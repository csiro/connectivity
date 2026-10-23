use anyhow::{ensure, Context, Result};
use cubecl::Device;
use objc2_foundation::NSProcessInfo;

/// Initialize the native Metal runtime; routing logic is shared with CUDA in gpu.rs.
pub(crate) fn device() -> Result<Device> {
    // CubeCL's native Metal compiler uses MSL 3.2 and macOS 15 math settings.
    ensure!(
        NSProcessInfo::processInfo()
            .operatingSystemVersion()
            .majorVersion
            >= 15,
        "Metal GPU execution requires macOS 15 or newer; use device=\"cpu\""
    );
    Device::metal(Default::default()).context("Metal GPU is unavailable; use device=\"cpu\"")
}
