//! Run logs name the version of the connectivity-core engine, which Cargo.lock pins.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!(
        "cargo:rustc-env=CONNECTIVITY_CORE_VERSION={}",
        locked_version("connectivity-core").unwrap_or_else(|| "unknown".into())
    );
}

/// The version of `package` in the lock file, when there is one.
fn locked_version(package: &str) -> Option<String> {
    let lock = std::fs::read_to_string("Cargo.lock").ok()?;
    let name = format!("name = \"{package}\"");
    lock.split("[[package]]").find_map(|block| {
        let mut lines = block.lines().map(str::trim);
        lines.any(|line| line == name).then_some(())?;
        lines
            .find_map(|line| line.strip_prefix("version = \""))
            .and_then(|rest| rest.strip_suffix('"'))
            .map(str::to_owned)
    })
}
