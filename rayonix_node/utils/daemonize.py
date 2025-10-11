"""
Daemonization utilities for RAYONIX blockchain node - Production Ready
Comprehensive daemonization with proper process management and security
"""

import os
import sys
import time
import signal
import atexit
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import errno
import resource

logger = logging.getLogger("rayonix_daemonize")

class DaemonizationError(Exception):
    """Custom exception for daemonization errors"""
    pass

class DaemonContext:
    """
    Production daemon context manager with comprehensive process management
    Based on PEP 3143 with enhancements for production use
    """
    
    def __init__(
        self,
        pidfile: Optional[str] = None,
        stdin: str = '/dev/null',
        stdout: str = '/dev/null', 
        stderr: str = '/dev/null',
        working_directory: str = '/',
        umask: int = 0o022,
        signal_map: Optional[Dict[int, Callable]] = None,
        files_preserve: Optional[list] = None,
        detach_process: bool = True,
        prevent_core: bool = True,
        signal_timeout: int = 30
    ):
        self.pidfile = pidfile
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.working_directory = working_directory
        self.umask = umask
        self.signal_map = signal_map or {}
        self.files_preserve = files_preserve or []
        self.detach_process = detach_process
        self.prevent_core = prevent_core
        self.signal_timeout = signal_timeout
        
        self.original_pid = os.getpid()
        self.daemon_pid = None
        self.is_daemon = False
        
    def __enter__(self):
        """Enter daemon context"""
        if self.detach_process:
            self._daemonize()
        else:
            self._setup_non_daemon()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit daemon context"""
        if exc_type:
            logger.error(f"Daemon context error: {exc_type.__name__}: {exc_val}")
        
        if self.detach_process and self.is_daemon:
            self._cleanup()
    
    def _setup_non_daemon(self):
        """Setup for non-daemon mode with proper signal handling"""
        logger.info("Running in non-daemon mode with signal handling")
        self._setup_signal_handlers()
        self._create_pidfile()
    
    def _daemonize(self):
        """Comprehensive daemonization process"""
        logger.info("Starting daemonization process...")
        
        try:
            # First fork - detach from parent process
            self._first_fork()
            
            # Decouple from parent environment
            self._create_new_session()
            
            # Second fork - ensure we can't acquire a controlling terminal
            self._second_fork()
            
            # Set process attributes
            self._set_process_attributes()
            
            # Redirect standard file descriptors
            self._redirect_standard_fds()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Create PID file
            self._create_pidfile()
            
            # Prevent core dumps if requested
            if self.prevent_core:
                self._prevent_core_dumps()
            
            # Log successful daemonization
            self.daemon_pid = os.getpid()
            self.is_daemon = True
            logger.info(f"Daemonization complete. PID: {self.daemon_pid}")
            
        except Exception as e:
            logger.error(f"Daemonization failed: {e}")
            raise DaemonizationError(f"Failed to daemonize process: {e}")
    
    def _first_fork(self):
        """Perform first fork to detach from parent"""
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process - wait a moment then exit
                logger.info(f"First fork completed, parent exiting. Child PID: {pid}")
                time.sleep(1)  # Give child time to setup
                os._exit(0)
                
        except OSError as e:
            raise DaemonizationError(f"First fork failed: {e}")
        
        # Child process continues
        os.setsid()  # Create new session
    
    def _create_new_session(self):
        """Create new session and process group"""
        try:
            # Create new session and process group
            os.setsid()
            
            # Set file creation mask
            os.umask(self.umask)
            
            # Change to working directory
            os.chdir(self.working_directory)
            
        except OSError as e:
            raise DaemonizationError(f"Failed to create new session: {e}")
    
    def _second_fork(self):
        """Perform second fork to prevent controlling terminal acquisition"""
        try:
            pid = os.fork()
            if pid > 0:
                # First child process exits
                logger.info(f"Second fork completed, first child exiting. Grandchild PID: {pid}")
                os._exit(0)
                
        except OSError as e:
            raise DaemonizationError(f"Second fork failed: {e}")
        
        # Grandchild process continues (the actual daemon)
    
    def _set_process_attributes(self):
        """Set process attributes for daemon operation"""
        try:
            # Set process name if available
            try:
                import setproctitle
                setproctitle.setproctitle("rayonixd")
            except ImportError:
                # setproctitle not available, continue without it
                pass
            
            # Set nice value to lower priority
            try:
                os.nice(10)  # Lower priority
            except OSError:
                pass  # Nice value change not critical
            
        except Exception as e:
            logger.warning(f"Could not set all process attributes: {e}")
    
    def _redirect_standard_fds(self):
        """Redirect standard file descriptors"""
        try:
            # Close all open file descriptors except preserved ones
            self._close_file_descriptors()
            
            # Redirect standard file descriptors
            self._redirect_stdin()
            self._redirect_stdout()
            self._redirect_stderr()
            
        except Exception as e:
            raise DaemonizationError(f"Failed to redirect file descriptors: {e}")
    
    def _close_file_descriptors(self):
        """Close all open file descriptors except preserved ones"""
        try:
            # Get maximum file descriptor
            maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            if maxfd == resource.RLIM_INFINITY:
                maxfd = 8192  # Reasonable default
            
            # Close all file descriptors except preserved ones
            preserve_fds = {fd.fileno() for fd in self.files_preserve}
            preserve_fds.update({0, 1, 2})  # Keep stdio until we redirect
            
            for fd in range(3, maxfd):
                if fd not in preserve_fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass  # FD wasn't open
                        
        except Exception as e:
            logger.warning(f"Could not close all file descriptors: {e}")
    
    def _redirect_stdin(self):
        """Redirect stdin from /dev/null"""
        try:
            with open(self.stdin, 'rb', 0) as f:
                os.dup2(f.fileno(), sys.stdin.fileno())
        except OSError as e:
            raise DaemonizationError(f"Failed to redirect stdin: {e}")
    
    def _redirect_stdout(self):
        """Redirect stdout to file"""
        try:
            with open(self.stdout, 'ab', 0) as f:
                os.dup2(f.fileno(), sys.stdout.fileno())
        except OSError as e:
            raise DaemonizationError(f"Failed to redirect stdout: {e}")
    
    def _redirect_stderr(self):
        """Redirect stderr to file"""
        try:
            with open(self.stderr, 'ab', 0) as f:
                os.dup2(f.fileno(), sys.stderr.fileno())
        except OSError as e:
            raise DaemonizationError(f"Failed to redirect stderr: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            # Default signal handlers
            default_handlers = {
                signal.SIGTERM: self._handle_shutdown_signal,
                signal.SIGINT: self._handle_shutdown_signal,
                signal.SIGQUIT: self._handle_shutdown_signal,
                signal.SIGHUP: signal.SIG_IGN,  # Ignore terminal hangups
            }
            
            # Update with custom handlers
            default_handlers.update(self.signal_map)
            
            # Register handlers
            for sig, handler in default_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (OSError, ValueError) as e:
                    logger.warning(f"Could not set handler for signal {sig}: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not setup all signal handlers: {e}")
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        
        # Import here to avoid circular imports
        from rayonix_node.utils.helpers2 import remove_pid_file
        
        # Remove PID file
        if self.pidfile:
            try:
                remove_pid_file(self.pidfile)
            except Exception as e:
                logger.error(f"Failed to remove PID file: {e}")
        
        # Exit gracefully
        sys.exit(0)
    
    def _create_pidfile(self):
        """Create PID file with locking and validation"""
        if not self.pidfile:
            return
            
        try:
            pidfile_path = Path(self.pidfile)
            pidfile_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if PID file already exists and process is running
            if pidfile_path.exists():
                try:
                    with open(pidfile_path, 'r') as f:
                        existing_pid = int(f.read().strip())
                    
                    # Check if process is still running
                    if self._is_process_running(existing_pid):
                        raise DaemonizationError(
                            f"PID file {self.pidfile} already exists and process {existing_pid} is running"
                        )
                    else:
                        logger.warning(f"Removing stale PID file for process {existing_pid}")
                        pidfile_path.unlink()
                        
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not read existing PID file: {e}")
                    pidfile_path.unlink()  # Remove corrupt PID file
            
            # Create new PID file
            with open(pidfile_path, 'w') as f:
                f.write(str(os.getpid()) + '\n')
            
            # Set secure permissions
            pidfile_path.chmod(0o644)
            
            # Register cleanup function
            atexit.register(self._remove_pidfile)
            
            logger.info(f"PID file created: {self.pidfile}")
            
        except Exception as e:
            raise DaemonizationError(f"Failed to create PID file: {e}")
    
    def _remove_pidfile(self):
        """Remove PID file during cleanup"""
        if self.pidfile and os.path.exists(self.pidfile):
            try:
                os.unlink(self.pidfile)
                logger.info(f"PID file removed: {self.pidfile}")
            except OSError as e:
                logger.error(f"Failed to remove PID file: {e}")
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running"""
        try:
            os.kill(pid, 0)  # Send signal 0 to check if process exists
            return True
        except OSError as err:
            if err.errno == errno.ESRCH:
                return False  # Process doesn't exist
            elif err.errno == errno.EPERM:
                return True   # Process exists but no permission to signal
            else:
                raise
    
    def _prevent_core_dumps(self):
        """Prevent core dumps for security"""
        try:
            # Set core file size limit to 0
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (OSError, ValueError) as e:
            logger.warning(f"Could not disable core dumps: {e}")
    
    def _cleanup(self):
        """Cleanup daemon resources"""
        self._remove_pidfile()

def daemonize_process(
    pidfile: Optional[str] = None,
    stdin: str = '/dev/null',
    stdout: str = '/dev/null',
    stderr: str = '/dev/null',
    working_directory: str = '/',
    umask: int = 0o022,
    detach: bool = True
) -> bool:
    """
    Daemonize the current process - production ready wrapper
    
    Args:
        pidfile: Path to PID file
        stdin: Path to stdin file
        stdout: Path to stdout file  
        stderr: Path to stderr file
        working_directory: Working directory for daemon
        umask: File creation mask
        detach: Whether to fully detach from terminal
    
    Returns:
        bool: True if daemonization successful
    """
    try:
        with DaemonContext(
            pidfile=pidfile,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            working_directory=working_directory,
            umask=umask,
            detach_process=detach
        ):
            # If we reach here in daemon mode, we're the daemon process
            if detach:
                logger.info("Process successfully daemonized")
            return True
            
    except DaemonizationError as e:
        logger.error(f"Daemonization failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during daemonization: {e}")
        return False

def is_daemon_running(pidfile: str) -> bool:
    """
    Check if daemon is already running
    
    Args:
        pidfile: Path to PID file to check
        
    Returns:
        bool: True if daemon is running
    """
    try:
        pidfile_path = Path(pidfile)
        if not pidfile_path.exists():
            return False
        
        with open(pidfile_path, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process exists
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            # Process doesn't exist, remove stale PID file
            pidfile_path.unlink(missing_ok=True)
            return False
            
    except Exception as e:
        logger.warning(f"Error checking daemon status: {e}")
        return False

def send_signal_to_daemon(pidfile: str, signal_num: int = signal.SIGTERM) -> bool:
    """
    Send signal to running daemon
    
    Args:
        pidfile: Path to PID file
        signal_num: Signal number to send
        
    Returns:
        bool: True if signal was sent successfully
    """
    try:
        if not is_daemon_running(pidfile):
            logger.error("Daemon is not running")
            return False
        
        with open(pidfile, 'r') as f:
            pid = int(f.read().strip())
        
        os.kill(pid, signal_num)
        logger.info(f"Signal {signal_num} sent to daemon process {pid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send signal to daemon: {e}")
        return False

class DaemonManager:
    """
    Comprehensive daemon process manager for production use
    """
    
    def __init__(self, name: str, pidfile: str, working_dir: str = '/'):
        self.name = name
        self.pidfile = pidfile
        self.working_dir = working_dir
        self.context: Optional[DaemonContext] = None
    
    def start(self, detach: bool = True) -> bool:
        """Start the daemon"""
        if is_daemon_running(self.pidfile):
            logger.error(f"{self.name} is already running")
            return False
        
        try:
            self.context = DaemonContext(
                pidfile=self.pidfile,
                working_directory=self.working_dir,
                detach_process=detach
            )
            
            with self.context:
                logger.info(f"{self.name} daemon started successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start {self.name} daemon: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the daemon"""
        return send_signal_to_daemon(self.pidfile, signal.SIGTERM)
    
    def restart(self) -> bool:
        """Restart the daemon"""
        if self.stop():
            # Wait a moment for graceful shutdown
            time.sleep(2)
            return self.start()
        return False
    
    def status(self) -> Dict[str, Any]:
        """Get daemon status"""
        running = is_daemon_running(self.pidfile)
        status_info = {
            'name': self.name,
            'running': running,
            'pidfile': self.pidfile
        }
        
        if running:
            try:
                with open(self.pidfile, 'r') as f:
                    pid = int(f.read().strip())
                status_info['pid'] = pid
            except Exception as e:
                status_info['error'] = str(e)
        
        return status_info

# Convenience function for common daemonization pattern
def create_blockchain_daemon(pidfile: str, data_dir: str) -> DaemonManager:
    """
    Create a pre-configured daemon manager for blockchain nodes
    
    Args:
        pidfile: PID file path
        data_dir: Data directory for the node
        
    Returns:
        DaemonManager: Configured daemon manager
    """
    # Ensure data directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup log files in data directory
    log_dir = Path(data_dir) / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    stdout_file = log_dir / 'daemon.stdout'
    stderr_file = log_dir / 'daemon.stderr'
    
    return DaemonManager(
        name="rayonixd",
        pidfile=pidfile,
        working_dir=data_dir
    )