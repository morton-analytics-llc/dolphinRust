//! Spawning external tools (SNAPHU, RAiDER) without leaking this process's files.
//!
//! libhdf5 (and GDAL through it) opens files without `O_CLOEXEC` and takes a
//! `flock` on each one. A `flock` belongs to the open file description, which a
//! child inherits across `fork`/`exec`, so a child spawned while an HDF5 file is
//! open would hold that file's lock until it exits, even after the parent closes
//! the file, and a later create or read-write open of it fails with `EAGAIN`
//! (#191, #214).

use std::process::Command;

/// Highest descriptor count considered when the soft limit is unbounded or
/// larger: Linux's default `nr_open` ceiling, and above macOS's default
/// `kern.maxfilesperproc`.
const DESCRIPTOR_CEILING: libc::c_int = 1 << 20;

/// Make `command`'s child close every descriptor above stdin/stdout/stderr when
/// it execs, so it inherits none of the files this process has open.
///
/// The descriptors are marked close-on-exec in the child (not closed), so the
/// standard library's exec-error reporting pipe keeps working. They are released
/// when the child's exec completes, which can be shortly after `spawn` returns.
pub fn exclude_inherited_descriptors(command: &mut Command) -> &mut Command {
    use std::os::unix::process::CommandExt;
    let descriptor_limit = open_descriptor_limit();
    // SAFETY: the closure runs in the forked child before exec and calls only
    // `fcntl`, which is async-signal-safe; it does not allocate or lock.
    unsafe {
        command.pre_exec(move || {
            (3..descriptor_limit).for_each(|descriptor| {
                // A number that is not open fails with EBADF, which is fine.
                libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC);
            });
            Ok(())
        })
    }
}

/// One past the highest descriptor number this process can hold, read in the
/// parent because `getrlimit` is not async-signal-safe.
fn open_descriptor_limit() -> libc::c_int {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes only into the struct it is given.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return DESCRIPTOR_CEILING;
    }
    libc::c_int::try_from(limit.rlim_cur)
        .map_or(DESCRIPTOR_CEILING, |soft| soft.min(DESCRIPTOR_CEILING))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Read;
    use std::process::{Child, Stdio};
    use std::time::{Duration, Instant};

    /// Open `path` the way libhdf5 does: no `O_CLOEXEC`, then an exclusive `flock`.
    fn open_locked_without_cloexec(path: &std::path::Path) -> libc::c_int {
        let name = CString::new(path.to_str().unwrap()).unwrap();
        let descriptor = unsafe { libc::open(name.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o600) };
        assert!(descriptor >= 0);
        assert_eq!(unsafe { libc::flock(descriptor, libc::LOCK_EX) }, 0);
        descriptor
    }

    /// Whether a fresh open of `path` can take the exclusive lock now.
    fn can_lock(path: &std::path::Path) -> bool {
        let file = std::fs::File::open(path).unwrap();
        let descriptor = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        unsafe { libc::flock(descriptor, libc::LOCK_EX | libc::LOCK_NB) == 0 }
    }

    /// Spawn a long-lived child and wait until its shell has exec'd.
    fn spawn_sleeper(command: &mut Command) -> Child {
        let mut child = command
            .args(["-c", "echo ready; exec sleep 30"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = [0_u8; 6];
        child
            .stdout
            .as_mut()
            .unwrap()
            .read_exact(&mut ready)
            .unwrap();
        child
    }

    /// Lock held by a child spawned while `path` was open, after the parent
    /// closed it: polled briefly, since close-on-exec releases at exec.
    fn lock_held_by_child(path: &std::path::Path, isolate: bool) -> bool {
        let descriptor = open_locked_without_cloexec(path);
        let mut command = Command::new("sh");
        if isolate {
            exclude_inherited_descriptors(&mut command);
        }
        let mut child = spawn_sleeper(&mut command);
        unsafe { libc::close(descriptor) };
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut held = true;
        while held && Instant::now() < deadline {
            held = !can_lock(path);
            std::thread::sleep(Duration::from_millis(10));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        held
    }

    /// #214: a child spawned through `exclude_inherited_descriptors` does not
    /// keep the lock of a file its parent had open, and one spawned without it
    /// does (so the check can see the leak it guards against).
    #[test]
    fn excluded_child_does_not_hold_parent_file_locks() {
        let path = std::env::temp_dir().join(format!(
            "dolphin_core_spawn_lock_{}.lock",
            std::process::id()
        ));
        assert!(
            lock_held_by_child(&path, false),
            "plain spawn should leak the lock"
        );
        assert!(
            !lock_held_by_child(&path, true),
            "isolated child kept the lock"
        );
        std::fs::remove_file(&path).unwrap();
    }
}
