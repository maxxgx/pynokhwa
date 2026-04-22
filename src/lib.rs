use std::sync::{atomic, Arc, Mutex, Weak};

use image::{ImageBuffer, Rgb};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        ApiBackend, CameraControl, CameraFormat, CameraIndex, ControlValueDescription, FrameFormat,
        RequestedFormat, RequestedFormatType,
    },
};
use parking_lot::FairMutex;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

static CAMERA_REGISTRY: Lazy<Mutex<HashMap<u32, Weak<CameraInternal>>>> =    Lazy::new(|| Mutex::new(HashMap::new()));

#[pyfunction]
pub fn query() -> PyResult<Vec<(u32, String, String, String)>> {
    println!("[pynokhwa] query() called — using Nokhwa backend");

    let devices = match nokhwa::query(ApiBackend::Auto) {
        Ok(val) => val,
        Err(error) => return Err(PyRuntimeError::new_err(error.to_string())),
    };

    let mut result = Vec::new();

    // Map by index for quick lookup
    let mut seen_indices = std::collections::HashSet::new();

    // Add devices normally found by nokhwa
    for device in devices.into_iter() {
        if let CameraIndex::Index(index) = *device.index() {
            result.push((
                index,
                device.human_name(),
                device.description().to_owned(),
                device.misc(),
            ));
            seen_indices.insert(index);
        }
    }

    Ok(result)
}

#[pyfunction]
pub fn check_can_use(index: u32) -> PyResult<bool> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use std::panic;

    {
        let reg = CAMERA_REGISTRY.lock().unwrap();
        if let Some(weak) = reg.get(&index) {
            if weak.upgrade().is_some() {
                return Ok(true);
            }
        }
    }

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let result = panic::catch_unwind(|| {
        let cam = nokhwa::Camera::new(CameraIndex::Index(index), format)?;
        drop(cam);
        Ok::<_, nokhwa::NokhwaError>(())
    });

    match result {
        Ok(Ok(_)) => {
            // println!("\t[pynokhwa] Camera {} opened successfully", index);
            Ok(true)
        }
        Ok(Err(err)) => {
            // println!("\t[pynokhwa] Failed to open camera {}: {:?}", index, err);
            Ok(false)
        }
        Err(_) => {
            // println!("\t[pynokhwa] Panic while opening camera {}!", index);
            Ok(false) // return False instead of crashing Python
        }
    }
}

#[pymodule]
fn pynokhwa<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    nokhwa::nokhwa_initialize(|_| {});
    m.add_function(wrap_pyfunction!(query, m)?)?;
    m.add_function(wrap_pyfunction!(check_can_use, m)?)?;
    m.add_class::<Camera>()?;
    m.add_class::<CamFormat>()?;
    m.add_class::<CamControl>()?;
    Ok(())
}

type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

#[derive(Clone)]
struct CameraInternal {
    camera: Arc<FairMutex<Option<nokhwa::Camera>>>,
    active_count: Arc<atomic::AtomicUsize>,
    running: Arc<atomic::AtomicBool>,                // NEW
    worker: Arc<FairMutex<Option<std::thread::JoinHandle<()>>>>, // NEW
    last_frame: Arc<FairMutex<Arc<Option<Image>>>>,
    last_err: Arc<FairMutex<Option<nokhwa::NokhwaError>>>,
}
impl CameraInternal {
    fn new(cam: nokhwa::Camera) -> CameraInternal {
        CameraInternal {
            camera: Arc::new(FairMutex::new(Some(cam))),
            active_count: Arc::new(atomic::AtomicUsize::new(0)),
            running: Arc::new(atomic::AtomicBool::new(false)),      // NEW
            worker: Arc::new(FairMutex::new(None)),                  // NEW
            last_frame: Arc::new(FairMutex::new(Arc::new(None))),
            last_err: Arc::new(FairMutex::new(None)),
        }
    }

    fn start(&self, format: CameraFormat) -> Result<(), nokhwa::NokhwaError> {
        // bump user count first
        self.active_count.fetch_add(1, atomic::Ordering::SeqCst);

        // only start the worker if not already running
        if self.running.swap(true, atomic::Ordering::SeqCst) == false {
            let active_count = Arc::clone(&self.active_count);
            let last_frame = Arc::clone(&self.last_frame);
            let last_err = Arc::clone(&self.last_err);
            let running = Arc::clone(&self.running);
            let camera = Arc::clone(&self.camera);

            let handle = std::thread::spawn(move || {
                // Configure + open on the worker thread
                {
                    let mut cam_guard = camera.lock();
                    if let Some(ref mut cam) = *cam_guard {
                        if let Err(err) = cam.set_camera_format(format).and(cam.open_stream()) {
                            *last_err.lock() = Some(err);
                            running.store(false, atomic::Ordering::SeqCst);
                            return;
                        }
                    } else {
                        eprintln!("[pynokhwa] Tried to start, but camera was closed!");
                        running.store(false, atomic::Ordering::SeqCst);
                        return;
                    }
                }

                let mut consecutive_timeouts = 0;
                while running.load(atomic::Ordering::Relaxed)
                    && active_count.load(atomic::Ordering::Relaxed) > 0
                {
                    let maybe_frame = {
                        let mut cam_guard = camera.lock();
                        if let Some(ref mut cam) = *cam_guard {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cam.frame()))
                                .unwrap_or_else(|_| Err(nokhwa::NokhwaError::GeneralError("Frame capture panic".into())))
                        } else {
                            break;
                        }
                    };

                    match maybe_frame {
                        Ok(frame) => {
                            consecutive_timeouts = 0;
                            if let Ok(img) = frame.decode_image::<RgbFormat>() {
                                let (w, h) = (img.width(), img.height());
                                let raw = img.into_raw();
                                if let Some(buf) = ImageBuffer::from_raw(w, h, raw) {
                                    *last_frame.lock() = Arc::new(Some(buf));
                                }
                            }
                        }
                        Err(err) => {
                            consecutive_timeouts += 1;
                            if consecutive_timeouts > 3 {
                                *last_err.lock() = Some(err);
                                break;
                            }
                        }
                    }
                }

                // Stop stream on the same thread that opened it
                {
                    let mut cam_guard = camera.lock();
                    if let Some(ref mut cam) = *cam_guard {
                        let _ = cam.stop_stream();
                    }
                }

                running.store(false, atomic::Ordering::SeqCst);
                // done
            });

            *self.worker.lock() = Some(handle);
            println!("[pynokhwa] worker thread started"); // fixed log
        } else {
            // Check if the requested format matches the current camera format that is already streaming.
            let (have_cam, matches, current_fmt_opt) = {
                let guard = self.camera.lock();
                match *guard {
                    Some(ref cam) => {
                        let current = cam.camera_format(); // nokhwa returns by value
                        let eq = current == format;
                        (true, eq, Some(current))
                    }
                    None => (false, false, None),
                }
            };

            if !have_cam {
                // No camera while `running == true` is an inconsistent state.
                self.active_count.fetch_sub(1, Ordering::SeqCst);
                return Err(nokhwa::NokhwaError::GeneralError(
                    "Camera is not available while marked as running".into(),
                ));
            }

            if !matches {
                self.active_count.fetch_sub(1, Ordering::SeqCst);
                let current_dbg = current_fmt_opt
                    .map(|f| format!("{:?}x{:?}@{:?}", f.width(), f.height(), f.frame_rate()))
                    .unwrap_or_else(|| "<unknown>".into());
                let requested_fmt = format!("{:?}x{:?}@{:?}", format.width(), format.height(), format.frame_rate());
                return Err(nokhwa::NokhwaError::GeneralError(format!(
                    "Camera is already streaming with a different format. Current: {}, Requested: {}",
                    current_dbg, requested_fmt
                )));
            }
        }

        Ok(())
    }

    fn close(&self) {
        println!("[pynokhwa] Closing camera (conditional)...");
        let remaining = self.active_count.fetch_sub(1, atomic::Ordering::SeqCst).saturating_sub(1);
        println!("[pynokhwa] Remaining active users = {}", remaining);

        if remaining == 0 {
            println!("[pynokhwa] Last active user — requesting worker shutdown.");
            self.running.store(false, atomic::Ordering::SeqCst);

            // Join worker before mutating camera state
            if let Some(handle) = self.worker.lock().take() {
                let _ = handle.join();
                println!("[pynokhwa] Worker joined.");
            }

            // Now it’s safe to clear buffers and release the camera
            {
                let mut cam_guard = self.camera.lock();
                // camera stream already stopped on worker; just drop it
                let _ = cam_guard.take();
            }
            *self.last_frame.lock() = Arc::new(None);
            *self.last_err.lock() = None;

            let mut reg = CAMERA_REGISTRY.lock().unwrap();
            reg.retain(|_, weak| weak.upgrade().is_some());
        } else {
            println!("[pynokhwa] Other users still streaming; keeping worker alive.");
        }
    }

    fn last_frame(&self) -> Arc<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>> {
        Arc::clone(&self.last_frame.lock())
    }

}

impl Drop for CameraInternal {
    fn drop(&mut self) {
        let remaining = self.active_count.load(atomic::Ordering::SeqCst);
        println!("[pynokhwa] Dropping CameraInternal — active users = {}", remaining);

        // Ensure shutdown if someone forgot to call close()
        self.running.store(false, atomic::Ordering::SeqCst);
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }

        // Best-effort stop & drop
        if let Some(mut cam) = self.camera.lock().take() {
            let _ = cam.stop_stream();
        }
        *self.last_frame.lock() = Arc::new(None);
        *self.last_err.lock() = None;

        let mut reg = CAMERA_REGISTRY.lock().unwrap();
        reg.retain(|_, weak| weak.upgrade().is_some());

        println!("[pynokhwa] CameraInternal dropped cleanly.");
    }
}
#[derive(Clone)]
#[pyclass]
struct CamFormat {
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    #[pyo3(get)]
    frame_rate: u32,
    format: FrameFormat,
}

#[pymethods]
impl CamFormat {
    #[getter]
    fn get_format(&self) -> String {
        match self.format {
            FrameFormat::MJPEG => "mjpeg".to_string(),
            FrameFormat::YUYV => "yuyv".to_string(),
            FrameFormat::GRAY => "gray".to_string(),
            FrameFormat::NV12 => "nv12".to_string(),
            FrameFormat::RAWRGB => "rawrgb".to_string(),
            FrameFormat::RAWBGR => "rawbgr".to_string(),
        }
    }
    //#[setter]
    fn set_format(&mut self, fmt: String) -> PyResult<()> {
        self.format = match fmt.as_str() {
            "mjpeg" => FrameFormat::MJPEG,
            "yuyv" => FrameFormat::YUYV,
            "gray" => FrameFormat::GRAY,
            "nv12" => FrameFormat::NV12,
            "rawrgb" => FrameFormat::RAWRGB,
            "rawbgr" => FrameFormat::RAWBGR,

            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported value (should be one of 'mjpeg', 'yuyv')",
                ))
            }
        };
        Ok(())
    }
}

impl From<CamFormat> for CameraFormat {
    fn from(fmt: CamFormat) -> CameraFormat {
        CameraFormat::new_from(fmt.width, fmt.height, fmt.format, fmt.frame_rate)
    }
}

impl From<CameraFormat> for CamFormat {
    fn from(fmt: CameraFormat) -> Self {
        CamFormat {
            width: fmt.width(),
            height: fmt.height(),
            format: fmt.format(),
            frame_rate: fmt.frame_rate(),
        }
    }
}

#[pyclass]
struct CamControl {
    cam: Weak<FairMutex<Option<nokhwa::Camera>>>,
    control: Mutex<CameraControl>,
}

#[pymethods]
impl CamControl {
    fn value_range(&self) -> (i64, i64, i64) {
        let control = self.control.lock().unwrap();
        let control_desc = control.description();
        match control_desc {
            ControlValueDescription::None => (0, 0, 0),

            ControlValueDescription::Integer { value, step, .. } => {
                // Single integer — return value as both min and max
                (*value, *value, *step)
            }

            ControlValueDescription::IntegerRange { min, max, step, .. } => {
                (*min, *max, *step)
            }

            ControlValueDescription::Float { value, step, .. } => {
                // Convert to i64 with rounding
                (*value as i64, *value as i64, *step as i64)
            }

            ControlValueDescription::FloatRange { min, max, step, .. } => {
                (*min as i64, *max as i64, *step as i64)
            }

            ControlValueDescription::Boolean { .. } => {
                // Boolean is always 0..1
                (0, 1, 1)
            }

            ControlValueDescription::String { .. } => {
                // No numeric range — fallback to 0,0,0
                (0, 0, 0)
            }

            ControlValueDescription::Bytes { value, .. } => {
                // Use length as a proxy range
                let len = value.len() as i64;
                (0, len, 1)
            }

            ControlValueDescription::KeyValuePair { key, value, .. } => {
                // Just return key/value as min/max
                (*key as i64, *value as i64, 1)
            }

            ControlValueDescription::Point { value, .. } => {
                // Use x as min, y as max (arbitrary but consistent)
                (value.0 as i64, value.1 as i64, 1)
            }

            ControlValueDescription::Enum { possible, .. } => {
                // Enumerations: 0..(N-1)
                if possible.is_empty() {
                    (0, 0, 1)
                } else {
                    (0, possible.len() as i64 - 1, 1)
                }
            }

            ControlValueDescription::RGB { max, .. } => {
                // Use max R as min and max G as max (arbitrary, but consistent)
                (0, max.0 as i64, 1)
            }
        }
    }
    fn set_value(&self, value: Option<i64>) -> PyResult<()> {
        let mut control = self.control.lock().unwrap();
        match self.cam.upgrade() {
            Some(cam) => match value {
                Some(value) => {
                    control.set_active(true);
                    let mut cam_guard = cam.lock();
                    let camera = cam_guard.as_mut()
                        .ok_or_else(|| PyRuntimeError::new_err("Camera not initialized"))?;

                    match camera.set_camera_control(
                        control.control(),
                        nokhwa::utils::ControlValueSetter::Integer(value),
                    ) {
                        Ok(_) => Ok(()),
                        Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
                    }
                }
                None => {
                    control.set_active(false);
                    Ok(())
                }
            },
            None => Err(PyRuntimeError::new_err(
                "Control is unusable as camera object has been dropped".to_string(),
            )),
        }
    }

    fn is_active(&self) -> bool {
        self.control.lock().unwrap().active()
    }
    fn set_active(&self, active: bool) -> PyResult<()> {
        self.control.lock().unwrap().set_active(active);
        Ok(())
    }

}

#[pyclass]
struct Camera {
    cam: Arc<CameraInternal>
}

#[pymethods]
impl Camera {
    #[new]
    fn new(index: u32) -> PyResult<Camera> {
        // Step 1: Check registry for existing
        {
            let reg = CAMERA_REGISTRY.lock().unwrap();
            if let Some(existing_weak) = reg.get(&index) {
                if let Some(existing_cam) = existing_weak.upgrade() {
                    println!("[pynokhwa] Reusing existing CameraInternal for index {}", index);
                    return Ok(Camera { cam: existing_cam });
                }
            }
        }

        // Step 2: Create new nokhwa camera
        let raw_cam = match nokhwa::Camera::new(
            CameraIndex::Index(index),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ) {
            Ok(c) => c,
            Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
        };

        // Step 3: Wrap in Arc and register Weak
        let internal = Arc::new(CameraInternal::new(raw_cam));
        {
            let mut reg = CAMERA_REGISTRY.lock().unwrap();
            reg.insert(index, Arc::downgrade(&internal));
        }

        Ok(Camera { cam: internal })
    }

    fn open(&self, format: CamFormat) -> PyResult<()> {
        if let Err(error) = self.cam.start(format.into()) {
            return Err(PyRuntimeError::new_err(error.to_string()));
        }
        // Todo
        let has_captured = Arc::new(atomic::AtomicBool::new(false));
        let _has_captured_clone = Arc::clone(&has_captured);
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        self.cam.close();
        Ok(())
    }

    fn info(&self) -> PyResult<String> {
        let mut camera = self.cam.camera.lock();
        let cam = camera.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Camera not initialized"))?;

        Ok(format!(
            "Selected format: {:?}",
            cam.camera_format()
        ))
    }

    fn get_formats(&self) -> PyResult<Vec<CamFormat>> {
        let mut camera = self.cam.camera.lock();
        let cam = camera.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Camera not initialized"))?;

        match cam.compatible_camera_formats() {
            Ok(formats) => Ok(formats.into_iter().map(|x| x.into()).collect()),
            Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
        }
    }

    fn poll_frame(&self, py: Python) -> PyResult<Option<(u32, u32, Py<PyBytes>)>> {
        match &*self.cam.last_frame() {
            Some(frame) => {
                let bytes = PyBytes::new_bound(py, frame.as_raw());
                Ok(Some((frame.width(), frame.height(), bytes.into())))
            }
            None => Ok(None),
        }
    }
    fn check_err(&self) -> PyResult<()> {
        match &*self.cam.last_err.lock() {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
    fn get_controls(&self) -> PyResult<Vec<(String, CamControl)>> {
        let mut camera = self.cam.camera.lock();
        let cam = camera.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Camera not initialized"))?;

        match cam.camera_controls_string() {
            Ok(list) => Ok(list
                .into_iter()
                .map(|(name, control)| {
                    (
                        name,
                        CamControl {
                            control: Mutex::new(control),
                            cam: Arc::downgrade(&self.cam.camera),
                        },
                    )
                })
                .collect()),
            Err(_err) => {
                Ok(Vec::new()) // Nothing supported
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::CameraInternal;
    use nokhwa::Camera;
    use nokhwa::{pixel_format::RgbFormat, query, utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType}};
    use std::io::Write;

    #[test]
    fn test_query_cameras() {
        let devices = query(ApiBackend::Auto)
            .expect("Failed to query cameras");
        println!("Found {} devices", devices.len());
        for d in &devices {
            println!("{:?}", d);
        }

        // not necessarily non-zero, but should not crash
        assert!(devices.len() >= 0);
    }

    #[test]
    fn test_capture_frame() {
        use std::fs::File;
        // Only run if at least one camera is present
        let mut cam = match Camera::new(
            CameraIndex::Index(3),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping: no camera available ({})", e);
                return;
            }
        };

        // let fmt = cam.compatible_camera_formats();
        cam.open_stream();


        let frame = cam.frame().expect("Failed to get frame");
        println!(
            "Captured frame: {} bytes",
            frame.buffer().len()
        );
        assert!(!frame.buffer().is_empty());

        let mut file = File::create("frame.raw").expect("Failed to create output file");
        file.write_all(frame.buffer()).expect("Failed to write frame data");
        println!("Frame bytes written to frame.raw");
    }

    #[test]
    fn test_live_view_window() {
        use nokhwa::{
            pixel_format::RgbFormat,
            utils::{CameraIndex, RequestedFormat, RequestedFormatType},
            Camera,
        };
        use minifb::{Key, Window, WindowOptions};

        // Open camera index 0
        let mut cam = match Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping: no camera available ({})", e);
                return;
            }
        };

        cam.open_stream().expect("Failed to open camera stream");

        // Grab one frame to get resolution
        let frame = cam.frame().expect("Failed to capture initial frame");
        let decoded = frame.decode_image::<RgbFormat>().expect("Failed to decode frame");
        let (width, height) = (decoded.width(), decoded.height());

        let mut window = Window::new(
            "Live Camera View - Press ESC to exit",
            width as usize,
            height as usize,
            WindowOptions::default(),
        )
            .expect("Failed to create window");

        println!("Streaming... Press ESC to exit.");

        while window.is_open() && !window.is_key_down(Key::Escape) {
            let frame = match cam.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Frame error: {}", e);
                    continue;
                }
            };

            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("Decode error: {}", e);
                    continue;
                }
            };

            // Convert RgbImage -> u32 buffer for minifb
            let mut buffer: Vec<u32> = Vec::with_capacity((width * height) as usize);
            for pixel in decoded.pixels() {
                let [r, g, b] = pixel.0;
                buffer.push(((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
            }

            window
                .update_with_buffer(&buffer, width as usize, height as usize)
                .expect("Failed to update window");
        }

        println!("Live view closed.");
    }

    #[test]
    fn test_live_view_window_with_wrapper() {
        use nokhwa::{
            pixel_format::RgbFormat,
            utils::{CameraIndex, RequestedFormat, RequestedFormatType},
            Camera,
        };
        use minifb::{Key, Window, WindowOptions};
        use std::time::Duration;

        // --- 1. Create a nokhwa camera ---
        let cam = match Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        ) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping: no camera available ({})", e);
                return;
            }
        };

        // --- 2. Wrap it in your CameraInternal ---
        let wrapper = CameraInternal::new(cam);

        // --- 3. Start streaming ---
        let format = wrapper.camera.lock().as_ref().unwrap().camera_format();
        wrapper.start(format).expect("Failed to start camera");

        // --- 4. Wait briefly to allow first frame to arrive ---
        std::thread::sleep(Duration::from_millis(200));

        // --- 5. Try to grab an initial frame to size the window ---
        let mut frame_opt = wrapper.last_frame();
        let frame = loop {
            if let Some(ref img) = *frame_opt {
                break img.clone();
            }
            std::thread::sleep(Duration::from_millis(50));
            frame_opt = wrapper.last_frame();
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        let mut window = Window::new(
            "Live Camera View - Press ESC to exit",
            width,
            height,
            WindowOptions::default(),
        )
            .expect("Failed to create window");



        println!("Streaming from CameraInternal... Press ESC to exit.");

        // --- 6. Live display loop ---
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let latest_frame = wrapper.last_frame();
            if let Some(ref img) = *latest_frame {
                // Convert ImageBuffer<Rgb<u8>> into a u32 buffer for minifb
                let mut buffer: Vec<u32> = Vec::with_capacity(width * height);
                for pixel in img.pixels() {
                    let [r, g, b] = pixel.0;
                    buffer.push(((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
                }

                if let Err(e) = window.update_with_buffer(&buffer, width, height) {
                    eprintln!("Failed to update window: {}", e);
                    break;
                }
            } else {
                println!("No frame yet...");
            }

            std::thread::sleep(Duration::from_millis(30)); // ~30 FPS
        }

        // --- 7. Shutdown ---
        println!("Shutting down...");
        wrapper.close();
    }

}
