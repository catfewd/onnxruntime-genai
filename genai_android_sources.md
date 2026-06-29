# ONNX Runtime GenAI Android Build Sources

# FILE: build.py

```
build.py
```

```py
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shlex
import shutil
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.append(str(REPO_ROOT / "tools" / "python"))
import util  # ./tools/python/util noqa: E402

log = util.get_logger("build.py")


def _path_from_env_var(env_var: str):
    env_var_value = os.environ.get(env_var)
    return Path(env_var_value) if env_var_value is not None else None


def _parse_args():
    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line):
            return shlex.split(arg_line)

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = Parser(
        description="ONNX Runtime GenAI Build Driver.",
        epilog=textwrap.dedent("""
            There are 3 phases which can be individually selected.

            The update (--update) phase will run CMake to generate makefiles.
            The build (--build) phase will build all projects.
            The test (--test) phase will run all unit tests.

            Default behavior is --update --build --test for native architecture builds.
            Default behavior is --update --build for cross-compiled builds.

            If phases are explicitly specified only those phases will be run.
            E.g., run with --build to rebuild without running the update or test phases.
            """),
        # files containing arguments can be specified on the command line with "@<filename>" and the arguments within
        # will be included at that point
        fromfile_prefix_chars="@",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--build_dir",
        type=Path,
        # We set the default programmatically as it needs to take into account whether we're cross-compiling
        help="Path to the build directory. Defaults to 'build/<target platform>'. "
        "The build configuration will be a subdirectory of the build directory. e.g. build/Linux/Debug",
    )

    parser.add_argument(
        "--config",
        default="RelWithDebInfo",
        type=str,
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration to build.",
    )

    # Build phases.
    parser.add_argument("--update", action="store_true", help="Update makefiles.")
    parser.add_argument("--build", action="store_true", help="Build.")
    parser.add_argument("--test", action="store_true", help="Run tests.")
    parser.add_argument("--package", action="store_true", help="Package the build.")  # Does not override other phases.
    parser.add_argument(
        "--clean", action="store_true", help="Run 'cmake --build --target clean' for the selected config."
    )

    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests. Overrides --test.")
    parser.add_argument("--skip_wheel", action="store_true", help="Skip building the Python wheel.")

    # Default to not building the language bindings
    parser.add_argument("--build_csharp", action="store_true", help="Build the C# API.")
    parser.add_argument("--build_java", action="store_true", help="Build Java bindings.")
    parser.add_argument(
        "--publish_java_maven_local",
        action="store_true",
        help="Publish Java bindings to local Maven repository after tests.",
    )

    parser.add_argument("--parallel", action="store_true", help="Enable parallel build.")

    # CI's sometimes explicitly set the path to the CMake and CTest executables.
    parser.add_argument("--cmake_path", default="cmake", type=Path, help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", type=Path, help="Path to the CTest program.")

    parser.add_argument(
        "--cmake_generator",
        choices=[
            "MinGW Makefiles",
            "Ninja",
            "NMake Makefiles",
            "Unix Makefiles",
            "Visual Studio 17 2022",
            "Visual Studio 18 2026",
            "Xcode",
        ],
        default=("Visual Studio 17 2022" if util.is_windows() else "Unix Makefiles"),
        help="Specify the generator that CMake invokes.",
    )
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="append",
        help="Extra definitions to pass to CMake during build system "
        "generation. These are just CMake -D options without the leading -D.",
    )

    parser.add_argument("--ort_home", default=None, type=Path, help="Root directory of onnxruntime.")

    parser.add_argument("--use_cuda", action="store_true", help="Whether to use CUDA. Default is to not use cuda.")
    parser.add_argument(
        "--cuda_home",
        type=Path,
        help="Path to CUDA home. Read from CUDA_HOME or CUDA_PATH environment variable if not specified."
        "Used when --use_cuda is specified.",
    )

    parser.add_argument(
        "--use_trt_rtx", action="store_true", help="Whether to use TensorRT-RTX. Default is to not use TensorRT-RTX."
    )

    parser.add_argument("--use_dml", action="store_true", help="Whether to use DML. Default is to not use DML.")

    parser.add_argument(
        "--use_winml", action="store_true", help="Whether to use WinML. Default is to not use WinML."
    )
    parser.add_argument(
        "--winml_sdk_version",
        type=str,
        default="2.1.1",
        help="Version of the Microsoft.Windows.AI.MachineLearning NuGet package to use. "
        "Only used when --use_winml is specified. Default is 2.1.1.",
    )

    parser.add_argument(
        "--use_guidance", action="store_true", help="Whether to add guidance support. Default is False."
    )

    # The following options are mutually exclusive (cross compiling options such as android, ios, etc.)
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("--android", action="store_true", help="Build for Android")
    platform_group.add_argument("--ios", action="store_true", help="Build for iOS")
    platform_group.add_argument(
        "--macos",
        choices=["MacOSX", "Catalyst"],
        help="Specify the target platform for macOS build. Only specify this argument when --build_apple_framework is present.",
    )

    # Android options
    parser.add_argument(
        "--android_abi",
        default="arm64-v8a",
        choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
        help="Specify the target Android Application Binary Interface (ABI)",
    )
    parser.add_argument(
        "--android_api", type=int, default=27, help="Android API Level. Default is 27 (Android 8.1, released in 2017)."
    )
    parser.add_argument(
        "--android_home", type=Path, default=_path_from_env_var("ANDROID_HOME"), help="Path to the Android SDK."
    )
    parser.add_argument(
        "--android_ndk_path",
        type=Path,
        default=_path_from_env_var("ANDROID_NDK_HOME"),
        help="Path to the Android NDK. Typically `<Android SDK>/ndk/<ndk_version>`.",
    )
    parser.add_argument(
        "--android_run_emulator",
        action="store_true",
        help="Create/start an Android emulator to run the test application. "
        "Requires --android, --build_java and --android_abi=x86_64.",
    )

    # iOS build options
    parser.add_argument(
        "--apple_sysroot",
        default="",
        help="Specify the location name of the macOS platform SDK to be used",
    )
    parser.add_argument(
        "--osx_arch",
        type=str,
        help="Specify the Target specific architectures for iOS This is only supported on MacOS host",
    )
    parser.add_argument(
        "--apple_deploy_target",
        type=str,
        help="Specify the minimum version of the target platform This is only supported on MacOS host",
    )

    parser.add_argument(
        "--build_apple_framework", action="store_true", help="Build a macOS/iOS framework for the ONNXRuntime."
    )

    parser.add_argument(
        "--arm64",
        action="store_true",
        help="[cross-compiling] Create ARM64 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )

    parser.add_argument(
        "--arm64ec",
        action="store_true",
        help="[cross-compiling] Create ARM64EC makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )

    parser.add_argument(
        "--skip_examples",
        action="store_true",
        help="Skip building the sample executables. Builds only on Linux and Windows otherwise.",
    )

    return parser.parse_args()


def _resolve_executable_path(command_or_path: Path, resolution_failure_allowed: bool = False):
    """
    Returns the absolute path of an executable.
    If `resolution_failure_allowed` is True, returns None if the executable path cannot be found.
    """
    executable_path = shutil.which(str(command_or_path))
    if executable_path is None:
        if resolution_failure_allowed:
            return None
        else:
            raise ValueError(f"Failed to resolve executable path for '{command_or_path}'.")

    return Path(executable_path)


def _validate_build_dir(args: argparse.Namespace):
    if not args.build_dir:
        target_sys = platform.system()

        # override if we're cross-compiling
        # TODO: Add ios and arm64 support
        if args.android:
            target_sys = "Android"
        elif platform.system() == "Darwin":
            # also tweak build directory name for mac builds
            target_sys = "macOS"

        args.build_dir = Path("build") / target_sys

    # set to a config specific build dir. it should exist unless we're creating the cmake setup
    is_strict = not args.update
    args.build_dir = args.build_dir.resolve(strict=is_strict) / args.config


def _validate_cuda_args(args: argparse.Namespace):
    if args.cuda_home and not args.use_trt_rtx:
        # default use_cuda to True if cuda_home is specified and use_trt_rtx is not enabled
        args.use_cuda = True

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else _path_from_env_var("CUDA_HOME")
        if not cuda_home and util.is_windows():
            cuda_home = _path_from_env_var("CUDA_PATH")

        cuda_home_valid = cuda_home.exists() if cuda_home else False

        if not cuda_home_valid:
            raise RuntimeError(
                f"cuda_home paths must be specified and valid. cuda_home='{cuda_home}' valid={cuda_home_valid}."
            )

        args.cuda_home = cuda_home.resolve(strict=True)


def _validate_trt_rtx_args(args: argparse.Namespace):
    if args.use_trt_rtx:
        # TRT_RTX requires CUDA, so validate cuda_home
        cuda_home = args.cuda_home if args.cuda_home else _path_from_env_var("CUDA_HOME")
        if not cuda_home and util.is_windows():
            cuda_home = _path_from_env_var("CUDA_PATH")

        cuda_home_valid = cuda_home.exists() if cuda_home else False

        if not cuda_home_valid:
            raise RuntimeError(
                f"TRT_RTX requires CUDA. cuda_home paths must be specified and valid. cuda_home='{cuda_home}' valid={cuda_home_valid}."
            )

        args.cuda_home = cuda_home.resolve(strict=True)


def _validate_winml_args(args: argparse.Namespace):
    if args.use_winml:
        if not util.is_windows():
            raise RuntimeError("--use_winml is only supported on Windows.")

        if not args.winml_sdk_version:
            # Fall back to the default so the CMake FATAL_ERROR for a missing
            # WINML_SDK_VERSION is never hit through the normal build.py flow.
            args.winml_sdk_version = "2.1.1"


def _validate_android_args(args: argparse.Namespace):
    if args.android:
        if not args.android_home:
            raise ValueError("--android_home is required to build for Android")

        if not args.android_ndk_path:
            raise ValueError("--android_ndk_path is required to build for Android")

        args.android_home = args.android_home.resolve(strict=True)
        args.android_ndk_path = args.android_ndk_path.resolve(strict=True)

        if not args.android_home.is_dir() or not args.android_ndk_path.is_dir():
            raise ValueError("Android home and NDK paths must be directories.")

        # auto-adjust the cmake generator for cross-compiling Android
        original_cmake_generator = args.cmake_generator
        if original_cmake_generator not in ["Ninja", "Unix Makefiles"]:
            if _resolve_executable_path("ninja", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Ninja"
            elif _resolve_executable_path("make", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Unix Makefiles"
            else:
                raise ValueError(
                    "Unable to find appropriate CMake generator for cross-compiling for Android. "
                    "Valid generators are 'Ninja' or 'Unix Makefiles'."
                )

        if args.cmake_generator != original_cmake_generator:
            log.info(f"Setting CMake generator to '{args.cmake_generator}' for cross-compiling for Android.")

        # no C# on Android so automatically skip
        args.build_csharp = False


def _validate_ios_args(args: argparse.Namespace):
    if args.ios:
        if not util.is_mac():
            raise ValueError("A Mac host is required to build for iOS")

        needed_args = [
            args.apple_sysroot,
            args.osx_arch,
            args.apple_deploy_target,
        ]
        arg_names = [
            "--apple_sysroot          <the location or name of the macOS platform SDK>",
            "--osx_arch              <the Target specific architectures for iOS>",
            "--apple_deploy_target   <the minimum version of the target platform>",
        ]
        have_required_args = all(_ is not None for _ in needed_args)
        if not have_required_args:
            raise ValueError(
                "iOS build on MacOS canceled due to missing arguments: "
                + ", ".join(val for val, cond in zip(arg_names, needed_args, strict=False) if not cond)
            )


def _validate_cmake_args(args: argparse.Namespace):
    args.cmake_extra_defines = [i for j in args.cmake_extra_defines for i in j] if args.cmake_extra_defines else []
    args.cmake_extra_defines = [f"-D{define}" for define in args.cmake_extra_defines]


def _validate_args(args: argparse.Namespace):
    # default to all 3 stages
    if not any((args.update, args.clean, args.build, args.test)):
        args.update = True
        args.build = True
        args.test = True

    # validate args. this updates values in args where applicable (e.g. fully resolve paths).
    args.cmake_path = _resolve_executable_path(args.cmake_path)
    args.ctest_path = _resolve_executable_path(args.ctest_path)

    _validate_build_dir(args)
    _validate_cuda_args(args)
    _validate_trt_rtx_args(args)
    _validate_winml_args(args)
    _validate_android_args(args)
    _validate_ios_args(args)
    _validate_cmake_args(args)

    if args.ort_home:
        if not args.ort_home.exists() or not args.ort_home.is_dir():
            raise ValueError(f"{args.ort_home} does not exist or is not a directory.")

        args.ort_home = args.ort_home.resolve(strict=True)


def _create_env(args: argparse.Namespace):
    env = os.environ.copy()

    if args.use_cuda or args.use_trt_rtx:
        env["CUDA_HOME"] = str(args.cuda_home)
        env["PATH"] = str(args.cuda_home / "bin") + os.pathsep + os.environ["PATH"]

    if args.android:
        env["ANDROID_HOME"] = str(args.android_home)
        env["ANDROID_NDK_HOME"] = str(args.android_ndk_path)

    return env


def _get_csharp_properties(args: argparse.Namespace, ort_lib_dir: Path | None = None):
    # Tests folder does not have a sln file. We use the csproj file to build and test.
    # The csproj file requires the platform to be AnyCPU (not "Any CPU")
    configuration = f"/p:Configuration={args.config}"
    platform = "/p:Platform=Any CPU"
    # need an extra config on windows as the actual build output is in the original build dir / config / config
    native_lib_path = (
        f"/p:NativeBuildOutputDir={str(args.build_dir / args.config) if util.is_windows() else str(args.build_dir)}"
    )

    if ort_lib_dir:
        ort_lib_path = f"/p:OrtLibDir={str(ort_lib_dir)}"
        props = [configuration, platform, native_lib_path, ort_lib_path]
    else:
        props = [configuration, platform, native_lib_path]

    return props


def _run_android_tests(args: argparse.Namespace):
    # only run the tests on the emulator for x86_64 currently.
    # TODO: may also be possible to run on a Mac with an arm64 chip
    if args.android_abi != "x86_64":
        log.info("Skipping Android tests as they are only supported on x86_64 currently.")
        return

    if not args.build_java:
        # currently we only have an Android test app that we run on the emulator to test the Java bindings.
        log.warning("Android testing requires --build_java to be set.")
        return

    sdk_tool_paths = util.android.get_sdk_tool_paths(args.android_home)
    adb = sdk_tool_paths.adb
    with contextlib.ExitStack() as context_stack:
        # use API 27 or higher so the emulator is Android 8.1 (2017) or later
        android_api = max(args.android_api, 27)

        if args.android_run_emulator:
            avd_name = "ort_genai_android"
            system_image = f"system-images;android-{android_api};default;{args.android_abi}"

            util.android.create_virtual_device(sdk_tool_paths, system_image, avd_name)
            emulator_proc = context_stack.enter_context(
                util.android.start_emulator(
                    sdk_tool_paths=sdk_tool_paths,
                    avd_name=avd_name,
                    extra_args=["-partition-size", "2047", "-wipe-data"],
                )
            )
            context_stack.callback(util.android.stop_emulator, emulator_proc)

        # use the gradle wrapper under <repo root>/java to run the test app on the emulator.
        # the test app loads and runs a test model using the GenAI Java bindings
        gradle_executable = str(REPO_ROOT / "src" / "java" / ("gradlew.bat" if util.is_windows() else "gradlew"))
        android_test_path = args.build_dir / "src" / "java" / "androidtest"
        import subprocess

        exception = None
        try:
            util.run(
                [gradle_executable, "--no-daemon", f"-DminSdkVer={android_api}", "clean", "connectedDebugAndroidTest"],
                cwd=android_test_path,
                capture_stdout=True,
                capture_stderr=True,
            )
        except subprocess.CalledProcessError as e:
            exception = e
            print(e)
            print(f"Output:\n{e.output.decode('utf-8')}")
            print(f"stderr:\n{e.stderr.decode('utf-8')}")

        # Print test log output so we can easily check that the test ran as expected
        util.run([adb, "logcat", "-s", "-d", "GenAI:V ORTGenAIAndroidTest:V TestRunner:V"])

        if exception:
            # uncomment if you need more logcat output in a CI
            # util.run([adb, "logcat", "-d", "*:E"])
            raise exception


def _get_windows_build_args(args: argparse.Namespace):
    win_args = [
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=/profile /DYNAMICBASE",
        "-DCMAKE_MODULE_LINKER_FLAGS_INIT=/profile /DYNAMICBASE",
        "-DCMAKE_SHARED_LINKER_FLAGS_INIT=/profile /DYNAMICBASE",
    ]
    cmake_c_flags = "/EHsc /Qspectre /MP /guard:cf /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000"
    if args.config == "Release":
        cmake_c_flags += " /O2 /Ob2 /DNDEBUG"
    elif args.config == "RelWithDebInfo":
        cmake_c_flags += " /O2 /Ob1 /DNDEBUG"
    elif args.config == "Debug":
        cmake_c_flags += " /Ob0 /Od /RTC1"
    win_args += [
        "-DCMAKE_C_FLAGS_INIT=" + cmake_c_flags,
        "-DCMAKE_CXX_FLAGS_INIT=" + cmake_c_flags,
    ]
    if args.use_cuda:
        win_args += [
            '-DCMAKE_CUDA_FLAGS_INIT=/DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 -Xcompiler=" /MP /guard:cf /Qspectre " -allow-unsupported-compiler',
        ]
    return win_args


def update(args: argparse.Namespace, env: dict[str, str]):
    """
    Update the cmake build files.
    """

    # build the cmake command to create/update the build files
    command = [str(args.cmake_path)]

    command += ["-G", args.cmake_generator]

    if util.is_windows():
        if args.cmake_generator == "Ninja":
            if args.use_cuda or args.use_trt_rtx:
                command += ["-DCUDA_TOOLKIT_ROOT_DIR=" + str(args.cuda_home)]

        elif args.cmake_generator.startswith("Visual Studio"):
            toolset_options = []

            is_x64_host = platform.machine() == "AMD64"
            if is_x64_host:
                pass

            if args.use_cuda or args.use_trt_rtx:
                toolset_options += ["cuda=" + str(args.cuda_home)]

            if toolset_options:
                command += ["-T", ",".join(toolset_options)]

    command += [f"-DCMAKE_BUILD_TYPE={args.config}"]

    build_wheel = "OFF" if args.skip_wheel else "ON"

    command += [
        "-S",
        str(REPO_ROOT),
        "-B",
        str(args.build_dir),
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DUSE_CUDA={'ON' if args.use_cuda else 'OFF'}",
        f"-DUSE_TRT_RTX={'ON' if args.use_trt_rtx else 'OFF'}",
        f"-DUSE_DML={'ON' if args.use_dml else 'OFF'}",
        f"-DUSE_WINML={'ON' if args.use_winml else 'OFF'}",
        f"-DENABLE_JAVA={'ON' if args.build_java else 'OFF'}",
        f"-DBUILD_WHEEL={build_wheel}",
        f"-DUSE_GUIDANCE={'ON' if args.use_guidance else 'OFF'}",
        f"-DPUBLISH_JAVA_MAVEN_LOCAL={'ON' if args.publish_java_maven_local else 'OFF'}",
    ]

    if args.ort_home:
        command += [f"-DORT_HOME={args.ort_home}"]

    if args.use_winml:
        command += [f"-DWINML_SDK_VERSION={args.winml_sdk_version}"]

    if args.use_cuda or args.use_trt_rtx:
        cuda_compiler = str(args.cuda_home / "bin" / "nvcc")
        command += [f"-DCMAKE_CUDA_COMPILER={cuda_compiler}"]

    if args.package and util.is_windows():
        command += _get_windows_build_args(args)

    if args.android:
        command += [
            "-DCMAKE_TOOLCHAIN_FILE="
            + str((args.android_ndk_path / "build" / "cmake" / "android.toolchain.cmake").resolve(strict=True)),
            f"-DANDROID_PLATFORM=android-{args.android_api}",
            f"-DANDROID_ABI={args.android_abi}",
            f"-DANDROID_MIN_SDK={args.android_api}",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_TESTS=OFF",
        ]

    if args.ios or args.macos:
        platform_name = "macabi" if args.macos == "Catalyst" else args.apple_sysroot
        command += [
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target,
            f"-DBUILD_APPLE_FRAMEWORK={'ON' if args.build_apple_framework else 'OFF'}",
            "-DPLATFORM_NAME=" + platform_name,
        ]

        if args.ios or args.macos == "Catalyst" or args.build_apple_framework:
            command += [
                "-DENABLE_PYTHON=OFF",
                "-DENABLE_TESTS=OFF",
                "-DENABLE_MODEL_BENCHMARK=OFF",
            ]

    if args.macos:
        command += [
            f"-DCMAKE_OSX_SYSROOT={args.apple_sysroot}",
            f"-DCMAKE_OSX_ARCHITECTURES={args.osx_arch}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={args.apple_deploy_target}",
        ]

    if args.ios:

        def _get_opencv_toolchain_file():
            if args.apple_sysroot == "iphoneos":
                return (
                    REPO_ROOT
                    / "cmake"
                    / "external"
                    / "opencv"
                    / "platforms"
                    / "iOS"
                    / "cmake"
                    / "Toolchains"
                    / "Toolchain-iPhoneOS_Xcode.cmake"
                )
            else:
                return (
                    REPO_ROOT
                    / "cmake"
                    / "external"
                    / "opencv"
                    / "platforms"
                    / "iOS"
                    / "cmake"
                    / "Toolchains"
                    / "Toolchain-iPhoneSimulator_Xcode.cmake"
                )

        command += [
            "-DCMAKE_SYSTEM_NAME=iOS",
            f"-DIOS_ARCH={args.osx_arch}",
            f"-DIPHONEOS_DEPLOYMENT_TARGET={args.apple_deploy_target}",
            # The following arguments are specific to the OpenCV toolchain file
            f"-DCMAKE_TOOLCHAIN_FILE={_get_opencv_toolchain_file()}",
        ]
        if args.use_guidance:
            command += ["-DRust_CARGO_TARGET=aarch64-apple-ios-sim"]

    if args.macos == "Catalyst":
        if args.cmake_generator == "Xcode":
            raise Exception("Xcode CMake generator ('--cmake_generator Xcode') doesn't support Mac Catalyst build.")

        macabi_target = f"{args.osx_arch}-apple-ios{args.apple_deploy_target}-macabi"
        command += [
            "-DCMAKE_CXX_COMPILER_TARGET=" + macabi_target,
            "-DCMAKE_C_COMPILER_TARGET=" + macabi_target,
            "-DCMAKE_CC_COMPILER_TARGET=" + macabi_target,
            f"-DCMAKE_CXX_FLAGS=--target={macabi_target}",
            f"-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            f"-DCMAKE_C_FLAGS=--target={macabi_target}",
            f"-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            f"-DCMAKE_CC_FLAGS=--target={macabi_target}",
            f"-DCMAKE_CC_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            "-DMAC_CATALYST=1",
        ]

    if args.cmake_generator.startswith("Visual Studio"):
        if args.arm64:
            command += ["-A", "ARM64"]
        elif args.arm64ec:
            command += ["-A", "ARM64EC"]
        elif args.use_winml:
            # WinML resolves its ONNX Runtime artifacts based on the generator
            # platform (see cmake/ortlib.cmake). For a default x64 build the
            # platform is otherwise left unset, so set it explicitly.
            command += ["-A", "x64"]

    if args.arm64 or args.arm64ec:
        if args.test:
            log.warning(
                "Cannot test on host build machine for cross-compiled ARM64 builds. Will skip test running after build."
            )
            args.test = False

    if args.cmake_extra_defines != []:
        command += args.cmake_extra_defines

    util.run(command, env=env)


def build(args: argparse.Namespace, env: dict[str, str]):
    """
    Build the targets.
    """

    make_command = [str(args.cmake_path), "--build", str(args.build_dir), "--config", args.config]

    if args.parallel:
        make_command.append("--parallel")

    util.run(make_command, env=env)

    if not args.skip_wheel:
        make_command += ["--target", "PyPackageBuild"]
        util.run(make_command, env=env)

    lib_dir = args.build_dir
    if util.is_windows():
        # On Windows, the library files are in a subdirectory named after the configuration (e.g. Debug, Release, etc.)
        lib_dir = lib_dir / args.config

    if not args.ort_home:
        _ = util.download_dependencies(args.use_cuda, args.use_dml, lib_dir)
    else:
        lib_dir = args.ort_home / "lib"

    if args.build_csharp:
        dotnet = str(_resolve_executable_path("dotnet"))

        # Build the library
        csharp_build_command = [
            dotnet,
            "build",
            ".",
        ]
        csharp_build_command += _get_csharp_properties(args)
        util.run(csharp_build_command, cwd=REPO_ROOT / "src" / "csharp")
        util.run(csharp_build_command, cwd=REPO_ROOT / "test" / "csharp")


def package(args: argparse.Namespace, env: dict[str, str]):
    """
    Package the build output with CMake targets.
    """
    make_command = [
        str(args.cmake_path),
        "--build",
        str(args.build_dir),
        "--config",
        args.config,
        "--target",
        "package",
    ]
    if args.parallel:
        make_command.append("--parallel")
    util.run(make_command, env=env)


def test(args: argparse.Namespace, env: dict[str, str]):
    """
    Run the tests.
    """
    lib_dir = args.build_dir
    if util.is_windows():
        # On Windows, the unit test executable is found inside a directory named after the configuration
        # (e.g. Debug, Release, etc.) within the test directory.
        # Whereas on as on platforms, the executable is directly under the test directory.
        lib_dir = lib_dir / args.config
    if not args.ort_home:
        _ = util.download_dependencies(args.use_cuda, args.use_dml, lib_dir)
    else:
        lib_dir = args.ort_home / "lib"

    ctest_cmd = [str(args.ctest_path), "--build-config", args.config, "--verbose", "--timeout", "10800"]
    util.run(ctest_cmd, cwd=str(args.build_dir))

    if args.build_csharp:
        dotnet = str(_resolve_executable_path("dotnet"))
        csharp_test_command = [dotnet, "test"]
        csharp_test_command += _get_csharp_properties(args, ort_lib_dir=lib_dir)
        util.run(csharp_test_command, env=env, cwd=str(REPO_ROOT / "test" / "csharp"))

    if args.build_java:
        ctest_cmd = [str(args.ctest_path), "--build-config", args.config, "--verbose", "--timeout", "10800"]
        util.run(ctest_cmd, cwd=str(args.build_dir / "src" / "java"))

    if args.android:
        _run_android_tests(args)


def clean(args: argparse.Namespace, env: dict[str, str]):
    """
    Clean the build output.
    """
    log.info("Cleaning targets")
    cmd_args = [str(args.cmake_path), "--build", str(args.build_dir), "--config", args.config, "--target", "clean"]
    util.run(cmd_args, env=env)


def build_examples(args: argparse.Namespace, env: dict[str, str]):
    """
    Build the examples.
    """
    examples_dir = REPO_ROOT / "examples" / "c"
    build_dir = examples_dir / "build"

    if build_dir.exists():
        log.info(f"Removing existing build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir()

    samples_to_build = [
        "-DMODEL_QA=ON",
        "-DMODEL_CHAT=ON",
        "-DMODEL_MM=ON",
        "-DWHISPER=ON",
        "-DNEMOTRON_SPEECH=ON"
    ]

    ort_include_dir = REPO_ROOT / "ort" / "include"
    ort_lib_dir = REPO_ROOT / "ort" / "lib"
    oga_include_dir = REPO_ROOT / "src"
    oga_lib_dir = args.build_dir
    if util.is_windows():
        # On Windows, the library files are in a subdirectory named after the configuration (e.g. Debug, Release, etc.)
        oga_lib_dir = oga_lib_dir / args.config

    cmake_command = (
        [
            str(args.cmake_path),
            "-S",
            str(examples_dir),
            "-B",
            str(build_dir),
            "-G",
            args.cmake_generator,
        ]
        + samples_to_build
        + [
            "-DORT_INCLUDE_DIR=" + str(ort_include_dir),
            "-DORT_LIB_DIR=" + str(ort_lib_dir),
            "-DOGA_INCLUDE_DIR=" + str(oga_include_dir),
            "-DOGA_LIB_DIR=" + str(oga_lib_dir),
        ]
    )

    if args.cmake_generator.startswith("Visual Studio"):
        if args.arm64:
            cmake_command += ["-A", "ARM64"]
        elif args.arm64ec:
            cmake_command += ["-A", "ARM64EC"]

    if args.cmake_extra_defines != []:
        cmake_command += args.cmake_extra_defines

    util.run(cmake_command, env=env)
    util.run([str(args.cmake_path), "--build", str(build_dir), "--config", args.config], env=env)


if __name__ == "__main__":
    if not (util.is_windows() or util.is_linux() or util.is_mac() or util.is_aix()):
        raise OSError(f"Unsupported platform {sys.platform}.")

    arguments = _parse_args()

    _validate_args(arguments)
    environment = _create_env(arguments)

    if arguments.update:
        update(arguments, environment)

    if arguments.clean:
        clean(arguments, environment)

    if arguments.build:
        build(arguments, environment)

    if arguments.package:
        package(arguments, environment)

    if arguments.test and not arguments.skip_tests:
        test(arguments, environment)

    if not (arguments.skip_examples or arguments.android or arguments.ios):
        build_examples(arguments, environment)
```


---

# FILE: tools/ci_build/github/android/build_aar_package.py

```
tools/ci_build/github/android/build_aar_package.py
```

```py
#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
BUILD_PY = REPO_ROOT / "build.py"
JAVA_ROOT = REPO_ROOT / "src" / "java"
sys.path.append(str(REPO_ROOT / "tools" / "python"))
import util  # REPO_ROOT/tools/python/util noqa: E402

# Default to only build 64-bit ABIs. TBD if we need to support any 32-bit ABIs.
# DEFAULT_BUILD_ABIS = ["armeabi-v7a", "arm64-v8a", "x86", "x86_64"]
DEFAULT_BUILD_ABIS = ["arm64-v8a", "x86_64"]

# Match onnxruntime and use NDK API 21 by default
# It is possible to build from source for Android API levels below 21, but it is not guaranteed.
# As level 21 is Android 5.0 from 2014, it's highly unlikely that we will need to support anything lower.
DEFAULT_ANDROID_MIN_SDK_VER = 21

# Android API 33+ is required for new apps and app updates from August 31, 2023
# Android API 34+ will be required from August 31, 2024
# See https://apilevels.com/
DEFAULT_ANDROID_TARGET_SDK_VER = 33


def _path_from_env_var(env_var: str):
    env_var_value = os.environ.get(env_var)
    return Path(env_var_value) if env_var_value is not None else None


def _parse_build_settings(args):
    setting_file = args.build_settings_file.resolve()

    if not setting_file.is_file():
        raise FileNotFoundError(f"Build config file {setting_file} is not a file.")

    with open(setting_file) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    if "build_abis" in build_settings_data:
        build_settings["build_abis"] = build_settings_data["build_abis"]
    else:
        build_settings["build_abis"] = DEFAULT_BUILD_ABIS

    build_params = []
    if "build_params" in build_settings_data:
        build_params += build_settings_data["build_params"]
    else:
        raise ValueError("build_params is required in the build config file")

    if "android_min_sdk_version" in build_settings_data:
        build_settings["android_min_sdk_version"] = build_settings_data["android_min_sdk_version"]
    else:
        build_settings["android_min_sdk_version"] = DEFAULT_ANDROID_MIN_SDK_VER

    build_params += ["--android_api=" + str(build_settings["android_min_sdk_version"])]

    if "android_target_sdk_version" in build_settings_data:
        build_settings["android_target_sdk_version"] = build_settings_data["android_target_sdk_version"]
    else:
        build_settings["android_target_sdk_version"] = DEFAULT_ANDROID_TARGET_SDK_VER

    if build_settings["android_min_sdk_version"] > build_settings["android_target_sdk_version"]:
        raise ValueError(
            f"android_min_sdk_version {build_settings['android_min_sdk_version']} cannot be larger than "
            f"android_target_sdk_version {build_settings['android_target_sdk_version']}"
        )

    build_settings["build_params"] = build_params

    return build_settings


def _build_aar(args):
    build_settings = _parse_build_settings(args)
    build_dir = Path(args.build_dir).resolve()

    # Setup temp environment for building
    temp_env = os.environ.copy()
    temp_env["ANDROID_HOME"] = str(args.android_home.resolve(strict=True))
    temp_env["ANDROID_NDK_HOME"] = str(args.android_ndk_path.resolve(strict=True))

    # Temp dirs to hold building results
    intermediates_dir = build_dir / "intermediates"
    build_config = args.config
    aar_dir = intermediates_dir / "aar" / build_config
    jnilibs_dir = intermediates_dir / "jnilibs" / build_config
    base_build_command = [sys.executable, str(BUILD_PY), f"--config={build_config}"]
    if args.ort_home:
        base_build_command += [f"--ort_home={args.ort_home!s}"]
    base_build_command += build_settings["build_params"]

    header_files_path = None

    # Build binary for each ABI, one by one
    for abi in build_settings["build_abis"]:
        abi_build_dir = intermediates_dir / abi
        abi_build_command = [*base_build_command, "--android_abi=" + abi, "--build_dir=" + str(abi_build_dir)]

        subprocess.run(abi_build_command, env=temp_env, shell=False, check=True, cwd=REPO_ROOT)

        # create symbolic links for libonnxruntime-genai.so and libonnxruntime-genai-jni.so
        # to jnilibs/[abi] for later compiling the aar package
        abi_jnilibs_dir = jnilibs_dir / abi
        abi_jnilibs_dir.mkdir(parents=True, exist_ok=True)
        for lib_name in ["libonnxruntime-genai.so", "libonnxruntime-genai-jni.so"]:
            src_lib_name = abi_build_dir / build_config / "src" / "java" / "android" / abi / lib_name
            target_lib_name = abi_jnilibs_dir / lib_name
            # If the symbolic already exists, delete it first
            target_lib_name.unlink(missing_ok=True)
            target_lib_name.symlink_to(src_lib_name)

        # we only need to define the header files path once
        if not header_files_path:
            header_files_path = abi_build_dir / build_config / "src" / "java" / "android" / "headers"

    # The directory to publish final AAR
    aar_publish_dir = os.path.join(build_dir, "aar_out", build_config)
    os.makedirs(aar_publish_dir, exist_ok=True)

    gradle_path = JAVA_ROOT / ("gradlew" if not util.is_windows() else "gradlew.bat")

    # get the common gradle command args
    gradle_command = [
        gradle_path,
        "--no-daemon",
        "-b=build-android.gradle",
        "-c=settings-android.gradle",
        f"-DjniLibsDir={jnilibs_dir}",
        f"-DbuildDir={aar_dir}",
        f"-DheadersDir={header_files_path}",
        f"-DpublishDir={aar_publish_dir}",
        f"-DminSdkVer={build_settings['android_min_sdk_version']}",
        f"-DtargetSdkVer={build_settings['android_target_sdk_version']}",
    ]

    # clean, build, and publish to a local directory
    subprocess.run([*gradle_command, "clean"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "build"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "publish"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Create Android Archive (AAR) package for one or more Android ABI(s)
        and building properties specified in the given build config file.
        See tools/ci_build/github/android/default_aar_build_settings.json for details.
        The output of the final AAR package can be found under [build_dir]/aar_out
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--android_home", type=Path, default=_path_from_env_var("ANDROID_HOME"), help="Path to the Android SDK."
    )

    parser.add_argument(
        "--android_ndk_path", type=Path, default=_path_from_env_var("ANDROID_NDK_HOME"), help="Path to the Android NDK."
    )

    parser.add_argument(
        "--build_dir",
        type=Path,
        default=(REPO_ROOT / "build" / "android_aar"),
        help="Provide the root directory for build output",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="Release",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration to build.",
    )

    parser.add_argument("--ort_home", type=Path, default=None, help="Path to an unzipped onnxruntime AAR.")

    parser.add_argument("build_settings_file", type=Path, help="Provide the file contains settings for building AAR")

    return parser.parse_args()


def main():
    args = parse_args()

    # Android SDK home and NDK path are required to come from explicit args or env vars.
    # If they're not set here, neither contained a value.
    if not args.android_home:
        raise ValueError("android_home is required")
    if not args.android_ndk_path:
        raise ValueError("android_ndk_path is required")

    _build_aar(args)


if __name__ == "__main__":
    main()
```


---

# FILE: tools/ci_build/github/android/default_aar_build_settings.json

```
tools/ci_build/github/android/default_aar_build_settings.json
```

```json
{
    "build_abis": [
        "arm64-v8a",
        "x86_64"
    ],
    "android_min_sdk_version": 24,
    "android_target_sdk_version": 33,
    "build_params": [
        "--android",
        "--parallel",
        "--cmake_generator=Ninja",
        "--build_java",
        "--skip_tests",
        "--skip_wheel"
    ]
}
```


---

# FILE: CMakeLists.txt

```
CMakeLists.txt
```

```txt
﻿# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # For xcframework support
  cmake_minimum_required(VERSION 3.28)
else()
  cmake_minimum_required(VERSION 3.26)
endif()

include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)

# All Options should be defined in cmake/options.cmake This must be included before any other cmake file is included
include(cmake/options.cmake)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11")
endif()

# Avoid warning of Calling FetchContent_Populate(Lib) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 OLD)
endif()

if(MSVC)
  # Add compiler options for security standards
  # Note: Must use separate generator expressions — do NOT use "C,CXX" in COMPILE_LANGUAGE.

  # Explicitly enable (/GS) stack protection to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/GS>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GS>")

  # Enable Control Flow Guard on compiler and linker to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/guard:cf>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>")
  add_link_options(/guard:cf /DYNAMICBASE)

  # Determine the target platform (from MSBuild generator)
  if (CMAKE_VS_PLATFORM_NAME)
    set(onnxruntime_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  # Enable CET shadow stack unless building for ARM64 (not supported there)
  if (onnxruntime_target_platform STREQUAL "x64" OR
      onnxruntime_target_platform STREQUAL "Win32" OR
      onnxruntime_target_platform STREQUAL "x86")
    add_link_options("/CETCOMPAT")
    message(STATUS "CET shadow stack enabled (/CETCOMPAT).")
  else()
    message(STATUS "CET shadow stack skipped.")
  endif()

  # Enable Spectre mitigations for C and C++ compilations only (avoid CUDA nvcc that is not supported there)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:/Qspectre>"
    "$<$<COMPILE_LANGUAGE:CXX>:/Qspectre>"
  )

  # Use updated value for __cplusplus macro (e.g. 201703L for C++17) instead of default 199711L
  # Required by some libraries/tools that check __cplusplus for feature support
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)

  add_compile_options(
    # Suppress warning C5038: data member will be initialized after another
    # Common when member initializer list is written out-of-order vs. declaration
    "$<$<COMPILE_LANGUAGE:C>:/w15038>"
    "$<$<COMPILE_LANGUAGE:CXX>:/w15038>"

    # Suppress warning C4100: unreferenced formal parameter
    # Often appears in template or virtual function overrides where a param is unused
    "$<$<COMPILE_LANGUAGE:C>:/wd4100>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4100>"

    # Suppress warning C4819: file contains character that cannot be represented in current code page
    "$<$<COMPILE_LANGUAGE:C>:/wd4819>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"

    # Suppress warning C4996: use of deprecated function or variable
    "$<$<COMPILE_LANGUAGE:C>:/wd4996>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"

    # Enable warning level 4 (more aggressive than default /W3)
    # Captures more potential bugs or code smells
    "$<$<COMPILE_LANGUAGE:C>:/W4>"
    "$<$<COMPILE_LANGUAGE:CXX>:/W4>"

    # Treat all warnings as errors (fail build on any warning)
    # Combine carefully with /W4 — suppress any known false positives
    "$<$<COMPILE_LANGUAGE:C>:/WX>"
    "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
  )
endif()

include(cmake/ortlib.cmake)


include(cmake/external/onnxruntime_external_deps.cmake)
# All Global variables, including GLOB, for the top level CMakeLists.txt should be defined here
include(cmake/global_variables.cmake)
# Checking if CUDA is supported
include(cmake/check_cuda.cmake)
# Checking if ROCm is supported
include(cmake/check_rocm.cmake)
# Checking if DML is supported
include(cmake/check_dml.cmake)

include(cmake/cxx_standard.cmake)

add_compile_definitions(BUILDING_ORT_GENAI_C)

add_compile_definitions(USE_GUIDANCE=$<BOOL:${USE_GUIDANCE}>)

# Suggested by https://gitlab.kitware.com/cmake/cmake/-/issues/20132
# MacCatalyst is not well supported in CMake
# The error that can emerge without this flag can look like:
# "clang : error : overriding '-mmacosx-version-min=11.0' option with '-target x86_64-apple-ios14.0-macabi' [-Werror,-Woverriding-t-option]"
if (PLATFORM_NAME STREQUAL "macabi")
  add_compile_options(-Wno-overriding-t-option)
  add_link_options(-Wno-overriding-t-option)
endif()

if(ENABLE_TESTS)
  # call enable_testing so we can add tests from subdirectories (e.g. test and src/java)
  # it applies recursively to all subdirectories
  enable_testing()
  if (TEST_PHI2)
    add_compile_definitions(TEST_PHI2=1)
  else()
    add_compile_definitions(TEST_PHI2=0)
  endif()

  if (TEST_QWEN_2_5)
    add_compile_definitions(TEST_QWEN_2_5=1)
  else()
    add_compile_definitions(TEST_QWEN_2_5=0)
  endif()

  if (USE_WEBGPU)
    add_compile_definitions(USE_WEBGPU=1)
  else()
    add_compile_definitions(USE_WEBGPU=0)
  endif()

endif()

if(ENABLE_TRACING)
  message(STATUS "Tracing is enabled.")
  add_compile_definitions(ORTGENAI_ENABLE_TRACING)
endif()

find_package(Threads REQUIRED)

if(WIN32)
  add_library(onnxruntime-genai SHARED ${generator_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MINOR=${VERSION_MINOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_PATCH=${VERSION_PATCH})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
  target_compile_definitions(onnxruntime-genai PRIVATE FILE_NAME=\"onnxruntime-genai.dll\")
else()
  add_library(onnxruntime-genai SHARED ${generator_srcs})
endif()

target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/shared/api)
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime_extensions)
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai PRIVATE Threads::Threads)

# The genai library itself is always embedded in the shared library
list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai>")

# we keep the shared libraries disconnected on Android as they will come from separate AARs and we don't want to force
# the ORT version to match in both.
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND (NOT BUILD_APPLE_FRAMEWORK) AND (NOT MAC_CATALYST)))
  add_compile_definitions(_ORT_GENAI_USE_DLOPEN)
else()
  target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})
  if(USE_WINML)
    target_link_options(onnxruntime-genai PRIVATE "/DELAYLOAD:${ONNXRUNTIME_LIB}")
  endif()
endif()

if(APPLE)
target_link_libraries(onnxruntime-genai PRIVATE "-framework Foundation" "-framework CoreML")
endif()


# Build all source files using CUDA as a separate shared library we dynamically load at runtime
if((USE_CUDA OR USE_TRT_RTX) AND CMAKE_CUDA_COMPILER)
  # Suppress nvcc warnings:
  # 1650 = "result of call is not used"
  # 221  = "floating-point value does not fit in required floating-point type"
  # Also suppress deprecated GPU targets warnings for CUDA 12.8+
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=1650>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=221>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4996>
  )

  # nvcc compiles CUDA code with cl.exe, so we need to pass the /Zc:preprocessor option to avoid compilation errors
  if(WIN32 AND MSVC AND MSVC_VERSION GREATER_EQUAL 1925)
    add_compile_options(
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/Zc:preprocessor>
    )
  endif()
  
  if(WIN32)
    add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_MINOR=${VERSION_MINOR})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_PATCH=${VERSION_PATCH})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE FILE_NAME=\"onnxruntime-genai-cuda.dll\")
  else()
    add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs})
  endif()
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${ORT_HEADER_DIR})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${GENERATORS_ROOT})
  target_link_libraries(onnxruntime-genai-cuda PRIVATE cublasLt cublas curand cufft cudart)
  set_target_properties(onnxruntime-genai-cuda PROPERTIES LINKER_LANGUAGE CUDA)
  add_dependencies(onnxruntime-genai onnxruntime-genai-cuda)
  source_group(TREE ${GENERATORS_ROOT}/cuda FILES ${generator_cudalib_srcs})
  list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai-cuda>")
  if(APPLE)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${GENERATORS_ROOT}/cuda/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${GENERATORS_ROOT}/cuda/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-DEF:\"${GENERATORS_ROOT}/cuda/symbols.def\"")
  else()
    message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
  endif()
endif()


if(USE_GUIDANCE)
  target_include_directories(onnxruntime-genai PUBLIC ${llguidance_SOURCE_DIR}/parser/)
  target_link_libraries(onnxruntime-genai PRIVATE llguidance)
  if (WIN32)
    # bcrypt is needed for the rust std lib
    target_link_libraries(onnxruntime-genai PRIVATE bcrypt)
  endif()
  if(MSVC)
    # The Rust llguidance static library is always compiled against the release MSVC CRT
    # (Rust has no debug CRT concept). The .lib embeds /DEFAULTLIB directives for the release
    # CRT and /NODEFAULTLIB directives that suppress the debug CRT (msvcrtd, ucrtd, vcruntimed).
    # In Debug builds, C++ code (e.g. onnxruntime-extensions) is compiled with /MDd and references
    # debug-only CRT functions like _CrtDbgReport (in ucrtd.lib). Because the Rust .lib suppresses
    # ucrtd.lib via its embedded /NODEFAULTLIB, _CrtDbgReport becomes unresolved.
    #
    # Fix: explicitly add the debug CRT import libraries in Debug builds. Explicitly specified
    # libraries are not affected by /NODEFAULTLIB directives (those only suppress /DEFAULTLIB
    # auto-linking). Also suppress the conflicting release CRT to avoid LNK4098 warnings.
    target_link_libraries(onnxruntime-genai PRIVATE
      $<$<CONFIG:Debug>:msvcrtd.lib>
      $<$<CONFIG:Debug>:ucrtd.lib>
      $<$<CONFIG:Debug>:vcruntimed.lib>
    )
    target_link_options(onnxruntime-genai PRIVATE
      $<$<CONFIG:Debug>:/NODEFAULTLIB:msvcrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:ucrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:vcruntime.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libcmt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libucrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libvcruntime.lib>
    )
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(onnxruntime-genai PUBLIC dl)  # For dlopen & co
endif()

if(USE_DML)
  list(APPEND ortgenai_embed_libs "${D3D12_LIB_DIR}/D3D12Core.dll")
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE ${DML_HEADER_DIR})
  target_include_directories(onnxruntime-genai PRIVATE ${D3D12_HEADER_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${DML_LIB_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${D3D12_LIB_DIR})
  target_link_libraries(onnxruntime-genai PRIVATE d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

  get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps ABSOLUTE)
  set(DXC_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.Direct3D.DXC.1.7.2308.12)
  set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/nuget.config)
  set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/packages.config)

  add_custom_command(
    OUTPUT
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
    DEPENDS
    ${PACKAGES_CONFIG}
    ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  add_custom_target(
    RESTORE_PACKAGES ALL
    DEPENDS
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
  )

  add_dependencies(RESTORE_PACKAGES nuget)
  add_dependencies(onnxruntime-genai RESTORE_PACKAGES)
endif()

if(ANDROID)
  # strip the binary if it's not a build with debug info
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_MINSIZEREL -s)

  # Build shared libraries with support for 16 KB page size on Android
  # https://source.android.com/docs/core/architecture/16kb-page-size/16kb#build-lib-16kb-alignment
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
endif()

if(ENABLE_TESTS)
  message("------------------Enabling tests------------------")
  add_subdirectory("${REPO_ROOT}/test")
endif()

if(ENABLE_PYTHON)
  message("------------------Enabling Python Wheel------------------")
  add_subdirectory("${SRC_ROOT}/python")
endif()

if (ENABLE_JAVA)
  message("------------------Enabling Java Jar------------------")
  add_subdirectory("${SRC_ROOT}/java")
endif()

if(ENABLE_MODEL_BENCHMARK)
  message("------------------Enabling model benchmark------------------")
  add_subdirectory("${REPO_ROOT}/benchmark/c")
endif()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

include(cmake/package.cmake)
```


---

# FILE: src/java/AndroidBuild.md

```
src/java/AndroidBuild.md
```

```md
# Android Build Setup

## Install Android SDK

Install the Android SDK and NDK. The ONNX Runtime instructions can be followed. 

https://onnxruntime.ai/docs/build/android.html#prerequisites

Use the latest release Android NDK available.


## Get the ONNX Runtime Android package

Download the ONNX Runtime Android package from Maven.

https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android

Unzip the AAR file to a directory. This directory will be specified as the `--ort_home` parameter.

## Build GenAI for Android with Java bindings.

Build from the root directory of the repository using build.bat or build.sh (which call build.py).

- Specify the Android SDK directory in the ANDROID_HOME environment variable, or `--android_home` parameter.
- Specify the Android NDK directory in the ANDROID_NDK_HOME environment variable, or `--android_ndk_path` parameter. 
  - This will be a subdirectory of the Android SDK's `ndk` directory.
- Specify the minimum Android API to build for in `--android_api`.
- Specify the ABI to build for in `--android_abi`. 
  - If testing using the emulator this should match the host machine architecture. 
  - If testing on an Android device this should match the device architecture, most likely `arm64-v8a`
- Specify `--ort_home` to be the path to the unzipped ONNX Runtime AAR. This directory should contain folders called `jni` and `headers`.
- On Windows, the `--cmake_generator` must be `Ninja`. 
  - This can be installed from [here](https://github.com/ninja-build/ninja/releases) or with  `pip install ninja`.

If `--android_run_emulator` is specified the build script will create and run the emulator. A unit test app will be deployed to the emulator and run.

Run the build script with `--help` for further details on all these options.

### Example build commands

#### Windows

.\build.bat --parallel  --config=Release --cmake_generator=Ninja --build_java --android --android_home=D:\Android --android_ndk_path=D:\Android\ndk\26.2.11394342 --android_api=27 --android_abi=x86_64 --ort_home=<path to unzipped onnxruntime-android.aar> --android_run_emulator

#### Linux

./build.sh --parallel  --config=Release --build_java --android --android_home=/home/me/Android --android_ndk_path=/home/me/Android/ndk/26.2.11394342 --android_api=27 --android_abi=x86_64 --ort_home=<path to unzipped onnxruntime-android.aar> --android_run_emulator

## Build an AAR with the GenAI Java bindings

There is also a script to build an AAR with x86_64 and arm64-v8a libraries. See /tools/ci_build/github/android/build_aar_package.py.

This AAR can be used in a test Android app. 
See src\java\src\test\android\app\build.gradle for example of how to manually specify onnxruntime-android and a custom built onnxruntime-genai AAR as dependencies in build.gradle. 
The dependencies can also be added to an app using Android Studio.
```


---

# FILE: src/java/CMakeLists.txt

```
src/java/CMakeLists.txt
```

```txt
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
include(FindJava)
find_package(Java REQUIRED)
include(UseJava)

if (NOT ANDROID)
  find_package(JNI REQUIRED)
endif()

set(JAVA_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
# <build output dir>/src/java (path used with add_subdirectory in root CMakeLists.txt)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Should we use onnxruntime-genai or onnxruntime-genai-static? Using onnxruntime-genai for now.
# Add dependency on native target
set(JAVA_DEPENDS onnxruntime-genai)

set(GRADLE_EXECUTABLE "${JAVA_SRC_ROOT}/gradlew")

file(GLOB_RECURSE genai4j_gradle_files "${JAVA_SRC_ROOT}/*.gradle")
file(GLOB_RECURSE genai4j_srcs "${JAVA_SRC_ROOT}/src/main/java/ai/onnxruntime-genai/*.java")

# set gradle options that are used with multiple gradle commands
if(WIN32)
  set(GRADLE_OPTIONS --console=plain -Dorg.gradle.daemon=false)
elseif (ANDROID)
  # For Android build, we may run gradle multiple times in same build. Sometimes gradle JVM will run out of memory
  # if we keep the daemon running, so we use no-daemon to avoid that
  set(GRADLE_OPTIONS --console=plain --no-daemon)
endif()

# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated
# and the onnxruntime-genai-jni target will be rebuilt
set(JAVA_OUTPUT_JAR ${JAVA_OUTPUT_DIR}/build/libs/onnxruntime-genai.jar)
set(GRADLE_ARGS clean jar -x test)

# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated
# and the onnxruntime-genai-jni target will be rebuilt
set(JAVA_OUTPUT_JAR ${JAVA_SRC_ROOT}/build/libs/onnxruntime-genai.jar)
set(GRADLE_ARGS clean jar -x test)

add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR}
                   COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_OPTIONS} ${GRADLE_ARGS}
                   WORKING_DIRECTORY ${JAVA_SRC_ROOT}
                   DEPENDS ${genai4j_gradle_files} ${genai4j_srcs})
add_custom_target(onnxruntime-genai4j DEPENDS ${JAVA_OUTPUT_JAR})

set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime-genai4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the JNI native sources
file(GLOB genai4j_native_src
    "${JAVA_SRC_ROOT}/src/main/native/*.cpp"
    "${JAVA_SRC_ROOT}/src/main/native/*.h"
    "${SRC_ROOT}/ort_genai_c.h"
    )

add_library(onnxruntime-genai-jni SHARED ${genai4j_native_src})
set_property(TARGET onnxruntime-genai-jni PROPERTY CXX_STANDARD 17)
add_dependencies(onnxruntime-genai-jni onnxruntime-genai4j)
# the JNI headers are generated in the genai4j target
target_include_directories(onnxruntime-genai-jni PRIVATE ${SRC_ROOT}
                                                           ${JAVA_SRC_ROOT}/build/headers
                                                           ${JNI_INCLUDE_DIRS})
target_link_libraries(onnxruntime-genai-jni PUBLIC onnxruntime-genai)

set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})

if (WIN32)
  set(JAVA_PLAT "win")
elseif (APPLE)
  set(JAVA_PLAT "osx")
elseif (LINUX)
  set(JAVA_PLAT "linux")
elseif (ANDROID)
  set(JAVA_PLAT "android")
else()
  message(FATAL_ERROR "GenAI with Java is not currently supported on this platform")
endif()

# Set platform and arch for packaging
if (genai_target_platform STREQUAL "x64")
  set(JNI_ARCH x64)
elseif (genai_target_platform STREQUAL "arm64")
  set(JNI_ARCH aarch64)
else()
  message(FATAL_ERROR "GenAI with Java is not currently supported on this platform")
endif()

# Similar to Nuget schema
set(JAVA_OS_ARCH ${JAVA_PLAT}-${JNI_ARCH})

# expose native libraries to the gradle build process
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/ai/onnxruntime/genai/native/${JAVA_OS_ARCH})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})

# Create directory for ONNX Runtime native libs
set(ORT_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/ai/onnxruntime/native/${JAVA_OS_ARCH})
file(MAKE_DIRECTORY ${ORT_PACKAGE_LIB_DIR})

# Add the native genai libraries to the native-lib dir
# Instead of hard-coding the .dll/.so name, we iterate over the list 
# of libraries the build system has identified as dependencies.
foreach(LIB_FILE ${ortgenai_embed_libs})
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${LIB_FILE}
                    ${JAVA_PACKAGE_LIB_DIR}/)
endforeach()

# Add the JNI bindings to the native-jni dir
add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      $<TARGET_FILE:onnxruntime-genai-jni>
                      ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime-genai-jni>)

if (${CMAKE_HOST_WIN32})
  # we need to use `call` with gradlew on Windows otherwise the POST_BUILD commands will be prematurely terminated
  # after the first gradlew command.
  set(POST_BUILD_GRADLE_EXECUTABLE "call" ${GRADLE_EXECUTABLE})
else()
  set(POST_BUILD_GRADLE_EXECUTABLE ${GRADLE_EXECUTABLE})
endif()

# run the build process
set(GRADLE_ARGS cmakeBuild)
if(PUBLISH_JAVA_MAVEN_LOCAL)
  list(APPEND GRADLE_ARGS publishToMavenLocal)
endif()
list(APPEND GRADLE_ARGS -DcmakeBuildDir=${JAVA_OUTPUT_DIR} -DnativeLibDir=${JAVA_NATIVE_LIB_DIR})
add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                   COMMAND ${POST_BUILD_GRADLE_EXECUTABLE} ${GRADLE_OPTIONS} ${GRADLE_ARGS}
                   WORKING_DIRECTORY ${JAVA_SRC_ROOT})

if (ANDROID)
  # <build dir>/src/java/build/android
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})

  # dirs to assemble the AAR contents in. <build dir>/src/java/android
  set(ANDROID_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/android)
  set(ANDROID_PACKAGE_HEADERS_DIR ${ANDROID_PACKAGE_DIR}/headers)
  set(ANDROID_PACKAGE_ABI_DIR ${ANDROID_PACKAGE_DIR}/${ANDROID_ABI})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_HEADERS_DIR})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_ABI_DIR})

  # copy C/C++ API headers to be packed into Android AAR package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_ROOT}/ort_genai.h ${ANDROID_PACKAGE_HEADERS_DIR})
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_ROOT}/ort_genai_c.h ${ANDROID_PACKAGE_HEADERS_DIR})

  # Copy onnxruntime-genai.so and onnxruntime-genai-jni.so for building Android AAR package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       $<TARGET_FILE:onnxruntime-genai>
                       ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime-genai>)

  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       $<TARGET_FILE:onnxruntime-genai-jni>
                       ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime-genai-jni>)

  # Generate the Android AAR package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E echo "Generating Android AAR package..."
                     COMMAND ${POST_BUILD_GRADLE_EXECUTABLE}
                       build
                       -b build-android.gradle -c settings-android.gradle
                       -DjniLibsDir=${ANDROID_PACKAGE_DIR} -DbuildDir=${ANDROID_PACKAGE_OUTPUT_DIR}
                       -DminSdkVer=${ANDROID_MIN_SDK} -DheadersDir=${ANDROID_PACKAGE_HEADERS_DIR}
                     WORKING_DIRECTORY ${JAVA_SRC_ROOT})

  # unit tests
  set(ANDROID_TEST_SRC_ROOT ${JAVA_SRC_ROOT}/src/test/android)
  set(ANDROID_TEST_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/androidtest)

  # copy the android test project to the build output so we can assemble all the pieces to build it
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
  file(GLOB android_test_files "${ANDROID_TEST_SRC_ROOT}/*")
  file(COPY ${android_test_files} DESTINATION ${ANDROID_TEST_PACKAGE_DIR})

  set(ANDROID_TEST_PACKAGE_LIB_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/libs)
  set(ANDROID_TEST_PACKAGE_APP_ASSETS_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/src/main/assets)
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_LIB_DIR})
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_APP_ASSETS_DIR})

  # Copy the test model to the assets folder in the test app
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                       ${REPO_ROOT}/test/models/hf-internal-testing/tiny-random-gpt2-fp32
                       ${ANDROID_TEST_PACKAGE_APP_ASSETS_DIR}/model)

  # Copy the Android AAR package we built to the libs folder of our test app
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${ANDROID_PACKAGE_OUTPUT_DIR}/outputs/aar/onnxruntime-genai-debug.aar
                       ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime-genai.aar)

  # Build Android test apk for java package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E echo "Building and running Android test for Android AAR package..."
                     COMMAND ${POST_BUILD_GRADLE_EXECUTABLE} clean assembleDebug assembleDebugAndroidTest
                       -DminSdkVer=${ANDROID_MIN_SDK} --stacktrace
                     WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
endif()

if (ENABLE_TESTS)
  message(STATUS "Adding Java tests")
  if (WIN32)
    # On windows ctest requires a test to be an .exe(.com) file
    # With gradle wrapper we get gradlew.bat. We delegate execution to a separate .cmake file
    # That can handle both .exe and .bat
    add_test(NAME onnxruntime-genai4j_test
             COMMAND ${CMAKE_COMMAND}
               -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE}
               -DBIN_DIR=${JAVA_OUTPUT_DIR}
               -DJAVA_SRC_ROOT=${JAVA_SRC_ROOT}
               -DJAVA_NATIVE_LIB_DIR=${JAVA_NATIVE_LIB_DIR}
               -P ${JAVA_SRC_ROOT}/windows-unittests.cmake)
  else()
    add_test(NAME onnxruntime-genai4j_test
             COMMAND ${GRADLE_EXECUTABLE} cmakeCheck
               -DcmakeBuildDir=${JAVA_OUTPUT_DIR} -DnativeLibDir=${JAVA_NATIVE_LIB_DIR}
             WORKING_DIRECTORY ${JAVA_SRC_ROOT})
  endif()

  if(WIN32)
    set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dll")
  elseif(APPLE)
    set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dylib")
  else()
    set(ONNXRUNTIME_GENAI_DEPENDENCY "*.so")
  endif()

  file(GLOB ort_native_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_GENAI_DEPENDENCY}")

  # Copy ORT native libs for Java tests
  foreach(LIB_FILE ${ort_native_libs})
    add_custom_command(
            TARGET onnxruntime-genai-jni POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                    ${LIB_FILE}
                    ${ORT_PACKAGE_LIB_DIR}/)
  endforeach()

  set_property(TEST onnxruntime-genai4j_test APPEND PROPERTY DEPENDS onnxruntime-genai-jni)
endif()
```


---

# FILE: src/java/Debugging.md

```
src/java/Debugging.md
```

```md
# Debugging Notes for Windows

## To debug using VS Code.

Create a config for java tests in settings.json. Adjust the paths based on your setup. 
My repo root is D:\src\github\ort.genai, and I was testing a Debug build on Windows. 

```yaml
    "java.test.config": {
        "testKind": "junit",
        "workingDirectory": "D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java",
        "classPaths": [ 
            "D:\\src\\github\\ort.genai\\src\\java\\build\\classes\\java\\main",
            "D:\\src\\github\\ort.genai\\src\\java\\build\\classes\\java\\test",
            "D:\\src\\github\\ort.genai\\src\\java\\build\\resources\\test"
        ],
        "sourcePaths": [
            "D:\\src\\github\\ort.genai\\src\\java\\src\\main\\java",
            "D:\\src\\github\\ort.genai\\src\\java\\src\\test\\java"
        ],
        "vmArgs": [ "-Djava.library.path=D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-lib\\ai\\onnxruntime-genai\\native\\win-x64" ],
    },
```

You may also want to set this in the VS Code settings: 
```yaml
    "java.debug.settings.onBuildFailureProceed": true,
```

I didn't try to setup VS Code to be able to build the tests using cmake or gradle, so the build VS Code attempts 
before run/debug of a test always fails. Instead I built the binding/test code from the command line, and then debugged
the tests from VS Code.

You can do a top level build (`./build --build_java --config Debug --build --test ...` from the repo root), 
or manually run gradlew from the src/java directory.

e.g. the gradlew command to build looks something like this. Adjust build output paths as needed.
> D:\src\github\ort.genai\src\java>./gradlew cmakeBuild '-DcmakeBuildDir=D:/src/github/ort.genai/build/Windows/Debug/src/java' '-DnativeLibDir=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime\genai\native\win-x64' '-Dorg.gradle.daemon=false' 

e.g. the gradlew command to test looks something like this. Adjust build output path as needed.
> D:\src\github\ort.genai\src\java>D:/src/github/ort.genai/src/java/gradlew --info cmakeCheck -DcmakeBuildDir="D:\src\github\ort.genai\build\Windows\Debug\src\java" -DnativeLibDir="D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -Dorg.gradle.daemon=false

Either of these commands can have ':spotlessApply' appended to them to automatically format the code as per the coding standards.

NOTE: If using the top-level build, the unit test code gets built in the 'test' phase - that's just how the gradle build is setup.

## To debug using IntelliJ

The test Debug/Run config for the test needs the cmakeBuildDir and nativeLibDir values to be added.

Easiest way to create a test config is to right-click on the test or test directory in the Project window and run it.
The run will fail, but the bulk of the configuration will be created for you.

Now open the test configuration, and in the 'Run' command which will start with something like
  `:test --tests "ai.onnxruntime.genai..."` 
add the values for cmakeBuildDir and nativeLibDir. 

e.g.
`"-DcmakeBuildDir=D:\src\github\ort.genai\build\Windows\Debug\src\java -DnativeLibDir=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64`

# Debugging native code

Download a junit-platform-console-standalone jar file from https://central.sonatype.com/artifact/org.junit.platform/junit-platform-console-standalone/versions

With that the magic incantation to run the tests from the command line (on Windows at least) is...

All tests:

> D:\Java\jdk-11.0.17\bin\java.exe "-Djava.library.path=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -jar junit-platform-console-standalone-1.10.2.jar -cp D:\src\github\ort.genai\src\java\build\classes\java\test -cp D:\src\github\ort.genai\src\java\build\resources\test  -cp D:\src\github\ort.genai\src\java\build\classes\java\main --scan-classpath

Specific test class uses `-c` and the full class name. e.g.

> D:\Java\jdk-11.0.17\bin\java.exe "-Djava.library.path=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -jar junit-platform-console-standalone-1.10.2.jar -cp D:\src\github\ort.genai\src\java\build\classes\java\test -cp D:\src\github\ort.genai\src\java\build\resources\test  -cp D:\src\github\ort.genai\src\java\build\classes\java\main -c ai.onnxruntime.genai.GenerationTest

Adjust the paths for your setup. Run from the java build output directory: e.g. D:\src\github\ort.genai\build\Windows\Debug\src\java 

That command can also be run from Visual Studio using the solution file for the native library you need to debug (onnxruntime-genai or onnxruntime) by setting the debug command, arguments and working directory in the project properties. 

e.g. to debug the `onnxruntime` project (which builds the onnxruntime shared library) in onnxruntime.sln, in the project properties for the `onnxruntime` project, under Debugging, set Command/Command Arguments/Working Directory to the above values. 
You can then right-click on the `onnxruntime` project -> Debug -> Start new instance. That should run java.exe and let you break on any exceptions with full symbols for the native code.

To also be able to set breakpoints, make sure a local debug build of the library is in the nativeLibDir so that java.exe is loading that.
```


---

# FILE: src/java/README.md

```
src/java/README.md
```

```md
# ONNX Runtime GenAI Java API

This directory contains the Java language binding for the ONNX Runtime GenAI.
Java Native Interface (JNI) is used to allow for seamless calls to ONNX Runtime GenAI from Java.

## Usage

This document pertains to developing, building, running, and testing the API itself in your local environment.
For general purpose usage of the publicly distributed API, please see the [general Java API documentation](https://www.onnxruntime.ai/docs/reference/api/java-api.html).

### Building

Build with the `--build_java` option.

Windows: `REPO_ROOT/build --build_java`
*nix: `REPO_ROOT/build.sh --build_java`

#### Requirements

Java 11 or later is required to build the library. The compiled jar file will run on Java 8 or later.

The [Gradle](https://gradle.org/) build system is used here to manage the Java project's dependency management, compilation, testing, and assembly.
In particular, the Gradle [wrapper](https://docs.gradle.org/current/userguide/gradle_wrapper.html) at `java/gradlew[.bat]` is used, locking the Gradle version to the one specified in the `java/gradle/wrapper/gradle-wrapper.properties` configuration.
Using the Gradle wrapper removes the need to have the right version of Gradle installed on the system.

#### Build Output

The build will generate output in `$REPO_ROOT/build/$OS/$CONFIGURATION/src/java`:

* `build/docs/javadoc/` - HTML javadoc
* `build/reports/` - detailed test results and other reports
* `build/libs/onnxruntime-genai-VERSION.jar` - JAR with compiled classes
* `native-jni` - platform-specific JNI shared library
* `native-lib` - platform-specific onnxruntime-genai and onnxruntime shared libraries.

#### Build System Overview

The main CMake build system delegates building and testing to Gradle.
This allows the CMake system to ensure all of the C/C++ compilation is achieved prior to the Java build.
The Java build depends on C/C++ onnxruntime-genai shared library and a C JNI shared library (source located in the `src/main/native` directory).
The JNI shared library is the glue that allows for Java to call functions in onnxruntime-genai shared library.
Given the fact that CMake injects native dependencies during CMake builds, some gradle tasks (primarily, `build`, `test`, and `check`) may fail.

When running the build script, CMake will compile the `onnxruntime-genai` target and the JNI glue `onnxruntime-genai-jni` target and expose the resulting libraries in a place where Gradle can ingest them.
Upon successful compilation of those targets, a special Gradle task to build will be executed. The results will be placed in the output directory stated above.

### Advanced Loading

The default behavior is to load the shared libraries using classpath resources.
If your use case requires custom loading of the shared libraries, please consult the javadoc in the [package-info.java](src/main/java/ai/onnxruntime-genai/package-info.java) or [OnnxRuntimeGenAI.java](src/main/java/ai/onnxruntime-genai/GenAI.java) files.

## Development

### Code Formatting

[Spotless](https://github.com/diffplug/spotless/tree/master/plugin-gradle) is used to keep the code properly formatted.
Gradle's `spotlessCheck` task will show any misformatted code.
Gradle's `spotlessApply` task will try to fix the formatting.
Misformatted code will raise failures when checks are ran during test run.

###  JNI Headers

When adding or updating native methods in the Java files, the auto-generated JNI headers in `build/headers/ai_onnxruntime-genai*.h` can be used to determine the JNI function signature.

These header files can be manually generated using Gradle's `compileJava` task which will compile the Java and update the header files accordingly.

Cut-and-paste the function declaration from the auto-generated .h file to add the implementation in the `./src/main/native/ai_onnxruntime-genai_*.cpp` file.

### Dependencies

The Java API does not have any runtime or compile dependencies.
```


---

# FILE: src/java/UpdatingJavaBindings.md

```
src/java/UpdatingJavaBindings.md
```

```md
# Updating Java Bindings

## Overview

The Java bindings expose the GenAI C API to Java. The starting point for determining new things to add is /src/ort_genai_c.h.
Guidance for how to add them is to look at how the python and C# bindings have been updated.
Keep things consistent across the language bindings in terms of classes and naming where possible.
It's probably easiest to compare to the C# classes. Pay attention to the C# class API and not so much about how they're implemented as the implementation may differ significantly across the languages in terms of types, memory management and error handling.

Note: 
- Directory names beginning with '/' are relative to the root of the repository.
- Directory names that don't begin with '/' are relative to the /src/java directory.

## Development Environment

VS Code, the free version of IntelliJ, Eclipse or whatever your Java IDE of choice is can be used. 
Load the /src/java directory as a project. 

Build/debug instructions are in Debugging.md for VS Code and IntelliJ. 

Note: It can be useful to copy ort_genai_c.h to the src/java/src/main/native directory temporarily. 
That at least lets VS Code provide intellisense for the C API when implementing JNI code as it can't determine the 
include path to find the header from the CMakeLists.txt file.

## Adding a new class
### Java
The Java class should be in a .java file named after the class in src/main/java/ai/onnxruntime/genai. 
Use existing classes as a guide.

### JNI C++ code
The Java class will have a matching .cpp file in src/main/native. 
The file name format is based on the Java class. 
e.g. ai_onnxruntime_genai_Generator.cc is the name for the JNI implementation of the Generator class in the ai.onnxruntime.genai package.

This is the format the Java build uses by default. Use the existing files as an example.

## Adding a new function
### Java
Add a new method to the relevant Java class and a new `native` function to call the GenAI C API function. 
- If the most sensible function name clashes with the Java class's method name, append 'Native' to the native function name
so there's no potential confusion about what the code is doing caused by having a class method and a native function with exactly the same names.
  - e.g. In the Generator class, 'generateNextToken' is the most sensible name at the Java and native level, so the Java function is called 'generateNextToken' and the native function is called 'generateNextTokenNative'.

Any addresses (e.g. for a newly created object) are passed between Java and JNI as a `long` and cast to the correct type.

The GenAI API will typically have a 'create' and 'destroy' function for the object being wrapped by the class.

Keep the code consistent with the existing classes.

The result of the 'create' is stored in a private member
- `private long nativeHandle = 0;`
The constructor should set this.
Other methods will pass this as the first argument to the native function.
- This is equivalent to using it as the `this` pointer of a C++ class.

Classes should be `AutoCloseable` and have a `close` method that calls the destroy function.

All classes should have a static init to ensure the native libraries are fully loaded upfront. This avoids cryptic errors.

Cut-and-paste this if adding a new class:
```java
  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }
```

### JNI C++ code
First compile the Java code with the `native` functions defined. No other implementation is needed at that point.
A header file for each class with the JNI function signatures will be generated in /src/java/build/headers.

To create/update the .cpp file that implements the JNI function (which will be located in src/main/native), 
cut-and-paste the relevant parts of the header into the .cpp file.

If creating the .cpp file, add a `#include` for the corresponding generated header file.
While not strictly necessary, doing so allows us to not have to explicitly specify the correct language linkage
(`extern "C"`) for the JNI functions as the linkage is inherited from the earlier declarations in the header.

Update the first 2 parameters of each new function to be meaningful by adding the parameter names
    Generated:  `JNIEnv*, jobject`
    Meaningful: `JNIEnv* env, jobject thiz`

The 'thiz' is the Java class instance that the function is being called on. 
It's rarely required in the JNI code as we're not trying to manipulate the Java object from the JNI level.

In the JNI native code function implementations the first step is to reinterpret_cast to the GenAI object type. 
Please keep things const correct (i.e. reinterpret_cast to a const pointer wherever possible).

There are helpers to check the OgaResult (`ThrowIfError`) and to throw exceptions (`ThrowException`). 
Exception handling is slightly counter-intuitive as you need to manually return from the JNI function. 
Calling `ThrowIfError` or `ThrowException` will create the exception that the Java level will receive, 
but the rest of the JNI function will execute before that happens.

That translates to 'if ThrowIfError or ThrowException are not the last thing in the function, you must check the return value (if ThrowIfError) and call `return ...;` if an exception was created.

The `CString` class provides a helper for converting between Java and C++ strings and managing memory.
This greatly simplifies the JNI code where strings are involved.

With all of the above, see existing code for examples.

## Testing

All new code should be covered by unit tests.
The tests should validate that the bindings work and handle expected potential misuse (e.g. invalid arguments) 
and edge cases that should be handled in the Java or JNI code. 
Validating the correctness of the GenAI API implementation is not the responsibility of the Java bindings unit tests.


## Formatting

https://github.com/microsoft/onnxruntime/blob/main/docs/Coding_Conventions_and_Standards.md
https://google.github.io/styleguide/javaguide.html

Easiest way to format the Java code is to run the gradlew task with `:spotlessApply`. See Debugging.md for the full commands.
Easiest way to format the C++ code is to use clang-format with /.clang-format.
```


---

# FILE: src/java/build-android.gradle

```
src/java/build-android.gradle
```

```gradle
apply plugin: 'com.android.library'
apply plugin: 'maven-publish'

def jniLibsDir = System.properties['jniLibsDir']
def buildDir = System.properties['buildDir']
def headersDir = System.properties['headersDir']
def publishDir = System.properties['publishDir']
def minSdkVer = System.properties['minSdkVer']
def targetSdkVer = System.properties['targetSdkVer']

// Since Android requires a higher numbers indicating more recent versions
// This function assumes the version number will be in format of A.B.C such as 1.7.0.
// Optional suffix of '-dev' is also handled
// We generate version code A[0{0,1}]B[0{0,1}]C,
// for example '1.7.0' -> 10700, '1.6.15' -> 10615
def static getVersionCode(String version){
	String[] codes = version.split('\\.');
	// Remove '-dev' suffix if it exists
	if (codes[2].contains('-')) {
		codes[2] = codes[2].split('-')[0];
	}

	// This will have problem if we have 3 digit [sub]version number, such as 1.7.199
	// but it is highly unlikely to happen
	String versionCodeStr = String.format("%d%02d%02d", codes[0] as int, codes[1] as int, codes[2] as int);
	return versionCodeStr as int;
}

project.buildDir = buildDir
project.version = rootProject.file('../../VERSION_INFO').text.trim()
project.group = "com.microsoft.onnxruntime"

def mavenArtifactId = project.name + '-android'
def defaultDescription = 'ONNX Runtime GenAI is ... ' +
	'This package contains the Android (AAR) build of ONNX Runtime GenAI, including Java bindings.'

buildscript {
	repositories {
		google()
		mavenCentral()
	}
	dependencies {
		classpath 'com.android.tools.build:gradle:7.4.2'

		// NOTE: Do not place your application dependencies here; they belong
		// in the individual module build.gradle files
	}
}

allprojects {
	repositories {
		google()
		mavenCentral()
	}
}

android {
	compileSdkVersion 32

	defaultConfig {
		minSdkVersion minSdkVer
		targetSdkVersion targetSdkVer
		versionCode = getVersionCode(project.version)
		versionName = project.version

		testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
	}

	android {
		lintOptions {
			abortOnError false
		}
	}

	buildTypes {
		release {
			minifyEnabled false
			debuggable false
			proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
		}
	}

	compileOptions {
		sourceCompatibility = JavaVersion.VERSION_1_8
		targetCompatibility = JavaVersion.VERSION_1_8
	}

	sourceSets {
		main {
			jniLibs.srcDirs = [jniLibsDir]
			java {
				srcDirs = ['src/main/java']
			}
		}
	}

	namespace 'ai.onnxruntime.genai'
}

task sourcesJar(type: Jar) {
	archiveClassifier = "sources"
	from android.sourceSets.main.java.srcDirs
}

task javadoc(type: Javadoc) {
	source = android.sourceSets.main.java.srcDirs
	classpath += project.files(android.getBootClasspath())
}

task javadocJar(type: Jar, dependsOn: javadoc) {
	archiveClassifier = 'javadoc'
	from javadoc.destinationDir
}

artifacts {
	archives javadocJar
	archives sourcesJar
}

dependencies {
	testImplementation 'org.junit.jupiter:junit-jupiter-api:5.7.0'
	testImplementation 'org.junit.platform:junit-platform-launcher:1.10.1'
	testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.7.0'
}

publishing {
	publications {
		maven(MavenPublication) {
			groupId = project.group
			artifactId = mavenArtifactId
			version = project.version

			// Three artifacts, the `aar`, the sources and the javadoc
			artifact("$buildDir/outputs/aar/${project.name}-release.aar")
			artifact javadocJar
			artifact sourcesJar

			pom {
				name = 'onnxruntime-genai'
				description = defaultDescription
				// TODO: Setup https://microsoft.github.io/onnxruntime-genai/ for equivalence with ORT?
				url = 'https://github.com/microsoft/onnxruntime-genai/'
				licenses {
					license {
						name = 'MIT License'
						url = 'https://opensource.org/licenses/MIT'
					}
				}
				organization {
					name = 'Microsoft'
					url = 'http://www.microsoft.com'
				}
				scm {
					connection = 'scm:git:git://github.com:microsoft/onnxruntime-genai.git'
					developerConnection = 'scm:git:ssh://github.com/microsoft/onnxruntime-genai.git'
					url = 'https://github.com/microsoft/onnxruntime-genai'
				}
				developers {
					// TODO: Does this need updating?
					developer {
						id = 'onnxruntime'
						name = 'ONNX Runtime'
						email = 'onnxruntime@microsoft.com'
					}
				}
			}
		}
	}

	//publish to filesystem repo
	repositories{
		maven {
			url "$publishDir"
		}
	}
}

// Add ORT C and C++ API headers to the AAR package, after task bundleDebugAar or bundleReleaseAar
// Such that developers using ORT native API can extract libraries and headers from AAR package without building ORT
tasks.whenTaskAdded { task ->
	if (task.name.startsWith("bundle") && task.name.endsWith("Aar")) {
		doLast {
			addFolderToAar("addHeadersTo" + task.name, task.archivePath, headersDir, 'headers')
		}
	}
}

def addFolderToAar(taskName, aarPath, folderPath, folderPathInAar) {
	def tmpDir = file("${buildDir}/${taskName}")
	tmpDir.mkdir()
	def tmpDirFolder = file("${tmpDir.path}/${folderPathInAar}")
	tmpDirFolder.mkdir()
	copy {
		from zipTree(aarPath)
		into tmpDir
	}
	copy {
		from fileTree(folderPath)
		into tmpDirFolder
	}
	ant.zip(destfile: aarPath) {
		fileset(dir: tmpDir.path)
	}
	delete tmpDir
}
```


---

# FILE: src/java/build.gradle

```
src/java/build.gradle
```

```gradle
buildscript {
	repositories {
		mavenCentral()
	}
	dependencies {
		classpath 'com.diffplug.spotless:spotless-plugin-gradle:6.25.0'
	}
}

plugins {
	id 'java-library'
	id 'maven-publish'
	id 'signing'
	id 'jacoco'
}

apply plugin: 'com.diffplug.spotless'

allprojects {
	repositories {
		mavenCentral()
	}
}

project.group = "com.microsoft.onnxruntime"
version = rootProject.file('../../VERSION_INFO').text.trim()
if (version.endsWith('-dev')) {
	version = version.replace('-dev', '-SNAPSHOT')
}

// cmake runs will inform us of the build directory of the current run
def cmakeBuildDir = System.properties['cmakeBuildDir']
def useCUDA = System.properties['USE_CUDA']
def cmakeJavaDir = "${cmakeBuildDir}"
def cmakeNativeLibDir = System.properties['nativeLibDir']
def cmakeBuildOutputDir = "${cmakeJavaDir}/build"

def mavenUser = System.properties['mavenUser']
def mavenPwd = System.properties['mavenPwd']
def mavenArtifactId = useCUDA == null ? project.name : project.name + "_gpu"

def onnxruntimeArtifactId = useCUDA == null ? "onnxruntime" : "onnxruntime_gpu"
def onnxruntimeVersion = System.properties["ONNXRUNTIME_VERSION"] ?: "1.23.0"

def defaultDescription = 'ONNX Runtime GenAI is <TODO>'

logger.lifecycle("cmakeBuildDir:${cmakeBuildDir}")
logger.lifecycle("cmakeNativeLibDir:${cmakeNativeLibDir}")

java {
	sourceCompatibility = JavaVersion.VERSION_1_8
	targetCompatibility = JavaVersion.VERSION_1_8
}

// This jar tasks serves as a CMAKE signalling
// mechanism. The jar will be overwritten by allJar task
jar {
}

// Add explicit sources jar with pom file.
task sourcesJar(type: Jar, dependsOn: classes) {
	archiveClassifier = "sources"
	from sourceSets.main.allSource
	into("META-INF/maven/$project.group/$mavenArtifactId") {
		from { generatePomFileForMavenPublication }
		rename ".*", "pom.xml"
	}
}

// Add explicit javadoc jar with pom file
task javadocJar(type: Jar, dependsOn: javadoc) {
	archiveClassifier = "javadoc"
	from javadoc.destinationDir
	into("META-INF/maven/$project.group/$mavenArtifactId") {
		from { generatePomFileForMavenPublication }
		rename ".*", "pom.xml"
	}
}

spotless {
	java {
		removeUnusedImports()
		googleJavaFormat()
	}
	format 'gradle', {
		target '**/*.gradle'
		trimTrailingWhitespace()
		indentWithTabs()

	}
}

compileJava {
	dependsOn spotlessJava
	options.compilerArgs += ["-h", "${project.buildDir}/headers/"]
	if (!JavaVersion.current().isJava8()) {
		// Ensures only methods present in Java 8 are used
		options.compilerArgs.addAll(['--release', '8'])
		// Gradle versions before 6.6 require that these flags are unset when using "-release"
		java.sourceCompatibility = null
		java.targetCompatibility = null
	}
}

compileTestJava {
	if (!JavaVersion.current().isJava8()) {
		// Ensures only methods present in Java 8 are used
		options.compilerArgs.addAll(['--release', '8'])
		// Gradle versions before 6.6 require that these flags are unset when using "-release"
		java.sourceCompatibility = null
		java.targetCompatibility = null
	}
}

sourceSets.main.java {
	srcDirs = ['src/main/java', 'src/main/jvm']
}

// expects
sourceSets.test {
	if (cmakeNativeLibDir != null) {
		def libDirs = cmakeNativeLibDir.split(File.pathSeparator)
		resources.srcDirs += libDirs
	}
}

if (cmakeNativeLibDir != null) {
	// generate tasks to be called from cmake

	// Overwrite jar location
	task allJar(type: Jar) {
		manifest {
			attributes('Automatic-Module-Name': "com.microsoft.onnxruntime.genai",
					'Implementation-Title': 'onnxruntime-genai',
					'Implementation-Version': project.version)
		}
		into("META-INF/maven/$project.group/$mavenArtifactId") {
			from { generatePomFileForMavenPublication }
			rename ".*", "pom.xml"
		}
		from sourceSets.main.output
		from cmakeNativeLibDir
	}

	task cmakeBuild(type: Copy) {
		from project.buildDir
		include 'libs/**'
		include 'docs/**'
		into cmakeBuildOutputDir
	}

	cmakeBuild.dependsOn allJar
	cmakeBuild.dependsOn sourcesJar
	cmakeBuild.dependsOn javadocJar
	cmakeBuild.dependsOn javadoc

	task cmakeCheck(type: Copy) {
		from project.buildDir
		include 'reports/**'
		into cmakeBuildOutputDir
	}
	cmakeCheck.dependsOn check
}

dependencies {
	implementation "com.microsoft.onnxruntime:${onnxruntimeArtifactId}:${onnxruntimeVersion}"
	testImplementation 'org.junit.jupiter:junit-jupiter-api:5.9.2'
	testImplementation 'org.junit.platform:junit-platform-launcher:1.10.1'
	testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.9.2'
}

processTestResources {
	duplicatesStrategy(DuplicatesStrategy.INCLUDE) // allows duplicates in the test resources
}

test {
	java {
		dependsOn spotlessJava
	}
	useJUnitPlatform()
	if (cmakeBuildDir != null) {
		workingDir cmakeBuildDir
	}

	systemProperty "java.library.path", cmakeNativeLibDir
	systemProperties System.getProperties().subMap(['USE_CUDA'])
	testLogging {
		events "passed", "skipped", "failed"
		showStandardStreams = true
		showStackTraces = true
		exceptionFormat = "full"
	}
}

jacocoTestReport {
	reports {
		xml.required = true
		csv.required = true
		html.destination file("${buildDir}/jacocoHtml")
	}
}

publishing {
	publications {
		maven(MavenPublication) {
			groupId = project.group
			artifactId = mavenArtifactId

			if (cmakeNativeLibDir != null) {
				artifact allJar
			} else {
				from components.java
			}
			artifact sourcesJar
			artifact javadocJar

			pom {
				name = 'onnxruntime-genai'
				description = defaultDescription
				// TODO: Setup https://microsoft.github.io/onnxruntime-genai/ for equivalence with ORT?
				url = 'https://github.com/microsoft/onnxruntime-genai/'
				licenses {
					license {
						name = 'MIT License'
						url = 'https://opensource.org/licenses/MIT'
					}
				}
				organization {
					name = 'Microsoft'
					url = 'https://www.microsoft.com'
				}
				scm {
					connection = 'scm:git:git://github.com:microsoft/onnxruntime-genai.git'
					developerConnection = 'scm:git:ssh://github.com/microsoft/onnxruntime-genai.git'
					url = 'https://github.com/microsoft/onnxruntime-genai'
				}
				developers {
					// TODO: Does this need updating?
					developer {
						id = 'onnxruntime'
						name = 'ONNX Runtime'
						email = 'onnxruntime@microsoft.com'
					}
				}
			}
		}
	}
	repositories {
		maven {
			name = 'sonatype'
			url 'https://oss.sonatype.org/service/local/staging/deploy/maven2/'
			credentials {
				username mavenUser
				password mavenPwd
			}
		}
	}
}

// Generates a task signMavenPublication that will
// build all artifacts.
signing {
	// Queries env vars:
	// ORG_GRADLE_PROJECT_signingKey
	// ORG_GRADLE_PROJECT_signingPassword but can be changed to properties
	def signingKey = findProperty("signingKey")
	def signingPassword = findProperty("signingPassword")
	if (signingKey && signingPassword) {
		useInMemoryPgpKeys(signingKey, signingPassword)
		sign publishing.publications.maven
	} else {
		logger.lifecycle("Signing key not found, skipping signing for local publication.")
	}
}
```


---

# FILE: src/java/gradle/wrapper/gradle-wrapper.properties

```
src/java/gradle/wrapper/gradle-wrapper.properties
```

```properties
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionSha256Sum=9631d53cf3e74bfa726893aee1f8994fee4e060c401335946dba2156f440f24c
distributionUrl=https\://services.gradle.org/distributions/gradle-8.6-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```


---

# FILE: src/java/gradlew

```
src/java/gradlew
```

```
#!/bin/sh

#
# Copyright © 2015-2021 the original authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

##############################################################################
#
#   Gradle start up script for POSIX generated by Gradle.
#
#   Important for running:
#
#   (1) You need a POSIX-compliant shell to run this script. If your /bin/sh is
#       noncompliant, but you have some other compliant shell such as ksh or
#       bash, then to run this script, type that shell name before the whole
#       command line, like:
#
#           ksh Gradle
#
#       Busybox and similar reduced shells will NOT work, because this script
#       requires all of these POSIX shell features:
#         * functions;
#         * expansions «$var», «${var}», «${var:-default}», «${var+SET}»,
#           «${var#prefix}», «${var%suffix}», and «$( cmd )»;
#         * compound commands having a testable exit status, especially «case»;
#         * various built-in commands including «command», «set», and «ulimit».
#
#   Important for patching:
#
#   (2) This script targets any POSIX shell, so it avoids extensions provided
#       by Bash, Ksh, etc; in particular arrays are avoided.
#
#       The "traditional" practice of packing multiple parameters into a
#       space-separated string is a well documented source of bugs and security
#       problems, so this is (mostly) avoided, by progressively accumulating
#       options in "$@", and eventually passing that to Java.
#
#       Where the inherited environment variables (DEFAULT_JVM_OPTS, JAVA_OPTS,
#       and GRADLE_OPTS) rely on word-splitting, this is performed explicitly;
#       see the in-line comments for details.
#
#       There are tweaks for specific operating systems such as AIX, CygWin,
#       Darwin, MinGW, and NonStop.
#
#   (3) This script is generated from the Groovy template
#       https://github.com/gradle/gradle/blob/HEAD/subprojects/plugins/src/main/resources/org/gradle/api/internal/plugins/unixStartScript.txt
#       within the Gradle project.
#
#       You can find Gradle at https://github.com/gradle/gradle/.
#
##############################################################################

# Attempt to set APP_HOME

# Resolve links: $0 may be a link
app_path=$0

# Need this for daisy-chained symlinks.
while
    APP_HOME=${app_path%"${app_path##*/}"}  # leaves a trailing /; empty if no leading path
    [ -h "$app_path" ]
do
    ls=$( ls -ld "$app_path" )
    link=${ls#*' -> '}
    case $link in             #(
      /*)   app_path=$link ;; #(
      *)    app_path=$APP_HOME$link ;;
    esac
done

# This is normally unused
# shellcheck disable=SC2034
APP_BASE_NAME=${0##*/}
# Discard cd standard output in case $CDPATH is set (https://github.com/gradle/gradle/issues/25036)
APP_HOME=$( cd "${APP_HOME:-./}" > /dev/null && pwd -P ) || exit

# Use the maximum available, or set MAX_FD != -1 to use that value.
MAX_FD=maximum

warn () {
    echo "$*"
} >&2

die () {
    echo
    echo "$*"
    echo
    exit 1
} >&2

# OS specific support (must be 'true' or 'false').
cygwin=false
msys=false
darwin=false
nonstop=false
case "$( uname )" in                #(
  CYGWIN* )         cygwin=true  ;; #(
  Darwin* )         darwin=true  ;; #(
  MSYS* | MINGW* )  msys=true    ;; #(
  NONSTOP* )        nonstop=true ;;
esac

CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar


# Determine the Java command to use to start the JVM.
if [ -n "$JAVA_HOME" ] ; then
    if [ -x "$JAVA_HOME/jre/sh/java" ] ; then
        # IBM's JDK on AIX uses strange locations for the executables
        JAVACMD=$JAVA_HOME/jre/sh/java
    else
        JAVACMD=$JAVA_HOME/bin/java
    fi
    if [ ! -x "$JAVACMD" ] ; then
        die "ERROR: JAVA_HOME is set to an invalid directory: $JAVA_HOME

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
    fi
else
    JAVACMD=java
    if ! command -v java >/dev/null 2>&1
    then
        die "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
    fi
fi

# Increase the maximum file descriptors if we can.
if ! "$cygwin" && ! "$darwin" && ! "$nonstop" ; then
    case $MAX_FD in #(
      max*)
        # In POSIX sh, ulimit -H is undefined. That's why the result is checked to see if it worked.
        # shellcheck disable=SC2039,SC3045
        MAX_FD=$( ulimit -H -n ) ||
            warn "Could not query maximum file descriptor limit"
    esac
    case $MAX_FD in  #(
      '' | soft) :;; #(
      *)
        # In POSIX sh, ulimit -n is undefined. That's why the result is checked to see if it worked.
        # shellcheck disable=SC2039,SC3045
        ulimit -n "$MAX_FD" ||
            warn "Could not set maximum file descriptor limit to $MAX_FD"
    esac
fi

# Collect all arguments for the java command, stacking in reverse order:
#   * args from the command line
#   * the main class name
#   * -classpath
#   * -D...appname settings
#   * --module-path (only if needed)
#   * DEFAULT_JVM_OPTS, JAVA_OPTS, and GRADLE_OPTS environment variables.

# For Cygwin or MSYS, switch paths to Windows format before running java
if "$cygwin" || "$msys" ; then
    APP_HOME=$( cygpath --path --mixed "$APP_HOME" )
    CLASSPATH=$( cygpath --path --mixed "$CLASSPATH" )

    JAVACMD=$( cygpath --unix "$JAVACMD" )

    # Now convert the arguments - kludge to limit ourselves to /bin/sh
    for arg do
        if
            case $arg in                                #(
              -*)   false ;;                            # don't mess with options #(
              /?*)  t=${arg#/} t=/${t%%/*}              # looks like a POSIX filepath
                    [ -e "$t" ] ;;                      #(
              *)    false ;;
            esac
        then
            arg=$( cygpath --path --ignore --mixed "$arg" )
        fi
        # Roll the args list around exactly as many times as the number of
        # args, so each arg winds up back in the position where it started, but
        # possibly modified.
        #
        # NB: a `for` loop captures its iteration list before it begins, so
        # changing the positional parameters here affects neither the number of
        # iterations, nor the values presented in `arg`.
        shift                   # remove old arg
        set -- "$@" "$arg"      # push replacement arg
    done
fi


# Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
DEFAULT_JVM_OPTS='"-Xmx64m" "-Xms64m"'

# Collect all arguments for the java command:
#   * DEFAULT_JVM_OPTS, JAVA_OPTS, JAVA_OPTS, and optsEnvironmentVar are not allowed to contain shell fragments,
#     and any embedded shellness will be escaped.
#   * For example: A user cannot expect ${Hostname} to be expanded, as it is an environment variable and will be
#     treated as '${Hostname}' itself on the command line.

set -- \
        "-Dorg.gradle.appname=$APP_BASE_NAME" \
        -classpath "$CLASSPATH" \
        org.gradle.wrapper.GradleWrapperMain \
        "$@"

# Stop when "xargs" is not available.
if ! command -v xargs >/dev/null 2>&1
then
    die "xargs is not available"
fi

# Use "xargs" to parse quoted args.
#
# With -n1 it outputs one arg per line, with the quotes and backslashes removed.
#
# In Bash we could simply go:
#
#   readarray ARGS < <( xargs -n1 <<<"$var" ) &&
#   set -- "${ARGS[@]}" "$@"
#
# but POSIX shell has neither arrays nor command substitution, so instead we
# post-process each arg (as a line of input to sed) to backslash-escape any
# character that might be a shell metacharacter, then use eval to reverse
# that process (while maintaining the separation between arguments), and wrap
# the whole thing up as a single "set" statement.
#
# This will of course break if any of these variables contains a newline or
# an unmatched quote.
#

eval "set -- $(
        printf '%s\n' "$DEFAULT_JVM_OPTS $JAVA_OPTS $GRADLE_OPTS" |
        xargs -n1 |
        sed ' s~[^-[:alnum:]+,./:=@_]~\\&~g; ' |
        tr '\n' ' '
    )" '"$@"'

exec "$JAVACMD" "$@"
```


---

# FILE: src/java/gradlew.bat

```
src/java/gradlew.bat
```

```bat
@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  Gradle startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
@rem This is normally unused
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo.
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo.
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar


@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=%APP_BASE_NAME%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd

:fail
rem Set variable GRADLE_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GRADLE_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
```


---

# FILE: src/java/settings-android.gradle

```
src/java/settings-android.gradle
```

```gradle
rootProject.name = 'onnxruntime-genai'
rootProject.buildFileName = 'build-android.gradle'
```


---

# FILE: src/java/settings.gradle

```
src/java/settings.gradle
```

```gradle
rootProject.name = 'onnxruntime-genai'
```


---

# FILE: src/java/src/main/AndroidManifest.xml

```
src/java/src/main/AndroidManifest.xml
```

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android" />
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Adapters.java

```
src/java/src/main/java/ai/onnxruntime/genai/Adapters.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** A container of adapters. */
public final class Adapters implements AutoCloseable {
  private long nativeHandle = 0;

  /**
   * Constructs an Adapters object with the given model.
   *
   * @param model The model.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Adapters(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalArgumentException("model has been freed and is invalid");
    }

    nativeHandle = createAdapters(model.nativeHandle());
  }

  /**
   * Loads the model adapter from the given adapter file path and adapter name.
   *
   * @param adapterFilePath The path of the adapter.
   * @param adapterName A unique user supplied adapter identifier.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void loadAdapter(String adapterFilePath, String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    loadAdapter(nativeHandle, adapterFilePath, adapterName);
  }

  /**
   * Unloads the adapter with the given identifier from the previosly loaded adapters. If the
   * adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.
   *
   * @param adapterName A unique user supplied adapter identifier.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void unloadAdapter(String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    unloadAdapter(nativeHandle, adapterName);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyAdapters(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createAdapters(long modelHandle) throws GenAIException;

  private native void destroyAdapters(long nativeHandle);

  private native void loadAdapter(long nativeHandle, String adapterFilePath, String adapterName)
      throws GenAIException;

  private native void unloadAdapter(long nativeHandle, String adapterName) throws GenAIException;
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Audios.java

```
src/java/src/main/java/ai/onnxruntime/genai/Audios.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** This class can load audios from the given path and prepare them for processing. */
public class Audios implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct an Audios instance.
   *
   * @param audioPath The audio path.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Audios(String audioPath) throws GenAIException {
    nativeHandle = loadAudios(audioPath);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyAudios(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long loadAudios(String audioPath) throws GenAIException;

  private native void destroyAudios(long audioshandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Config.java

```
src/java/src/main/java/ai/onnxruntime/genai/Config.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * Use Config to set the ORT execution providers (EPs) and their options. The EPs are applied based
 * on insertion order.
 */
public final class Config implements AutoCloseable {
  private long nativeHandle;

  /**
   * Creates a Config from the given configuration directory.
   *
   * @param modelPath The path to the configuration directory.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Config(String modelPath) throws GenAIException {
    nativeHandle = createConfig(modelPath);
  }

  /** Clear the list of providers in the config */
  public void clearProviders() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    clearProviders(nativeHandle);
  }

  /**
   * Add the provider at the end of the list of providers in the given config if it doesn't already
   * exist. If it already exists, does nothing.
   *
   * @param providerName The provider name.
   */
  public void appendProvider(String providerName) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    appendProvider(nativeHandle, providerName);
  }

  /**
   * Set a provider option.
   *
   * @param providerName The provider name.
   * @param optionKey The key of the option to set.
   * @param optionValue The value of the option to set.
   */
  public void setProviderOption(String providerName, String optionKey, String optionValue) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    setProviderOption(nativeHandle, providerName, optionKey, optionValue);
  }

  /**
   * Overlay JSON on top of the config file
   *
   * @param json The JSON string to overlay
   */
  public void overlay(String json) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    overlay(nativeHandle, json);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyConfig(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createConfig(String modelPath) throws GenAIException;

  private native void destroyConfig(long configHandle);

  private native void clearProviders(long configHandle);

  private native void appendProvider(long configHandle, String provider_name);

  private native void setProviderOption(
      long configHandle, String providerName, String optionKey, String optionValue);

  private native void overlay(long configHandle, String json);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/GenAI.java

```
src/java/src/main/java/ai/onnxruntime/genai/GenAI.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

final class GenAI {
  private static final Logger logger = Logger.getLogger(GenAI.class.getName());

  /**
   * The name of the system property which when set gives the path on disk where the ONNX Runtime
   * native libraries are stored.
   */
  static final String GENAI_NATIVE_PATH = "onnxruntime-genai.native.path";

  /** The short name of the ONNX Runtime GenAI shared library */
  static final String GENAI_LIBRARY_NAME = "onnxruntime-genai";

  /** The short name of the ONNX Runtime GenAI JNI shared library */
  static final String GENAI_JNI_LIBRARY_NAME = "onnxruntime-genai-jni";

  /** The short name of the ONNX runtime shared library */
  static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";

  static final String ONNXRUNTIME_GENAI_RESOURCE_PATH = "/ai/onnxruntime/genai/native/";
  static final String ONNXRUNTIME_RESOURCE_PATH = "/ai/onnxruntime/native/";

  /** The value of the GENAI_NATIVE_PATH system property */
  private static String libraryDirPathProperty;

  /** The OS & CPU architecture string */
  private static final String OS_ARCH_STR = initOsArch();

  /** Have the native libraries been loaded */
  private static boolean loaded = false;

  /** The temp directory where native libraries are extracted */
  private static Path tempDirectory;

  static synchronized void init() throws IOException {
    if (loaded) {
      return;
    }

    tempDirectory = isAndroid() ? null : Files.createTempDirectory("onnxruntime-genai-java");

    try {
      libraryDirPathProperty = System.getProperty(GENAI_NATIVE_PATH);

      load(ONNXRUNTIME_LIBRARY_NAME, ONNXRUNTIME_RESOURCE_PATH); // ORT native
      load(GENAI_LIBRARY_NAME, ONNXRUNTIME_GENAI_RESOURCE_PATH); // ORT GenAI native
      load(GENAI_JNI_LIBRARY_NAME, ONNXRUNTIME_GENAI_RESOURCE_PATH); // GenAI JNI layer
      loaded = true;
    } finally {
      if (tempDirectory != null) {
        cleanUp(tempDirectory.toFile());
      }
    }
  }

  static native void shutdown();

  /* Computes and initializes OS_ARCH_STR (such as linux-x64) */
  private static String initOsArch() {
    String detectedOS = null;
    String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
    if (os.contains("mac") || os.contains("darwin")) {
      detectedOS = "osx";
    } else if (os.contains("win")) {
      detectedOS = "win";
    } else if (os.contains("nux")) {
      detectedOS = "linux";
    } else if (isAndroid()) {
      detectedOS = "android";
    } else {
      throw new IllegalStateException("Unsupported os:" + os);
    }

    String detectedArch = null;
    String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
      detectedArch = "x64";
    } else if (arch.startsWith("x86")) {
      // 32-bit x86 is not supported by the Java API
      detectedArch = "x86";
    } else if (arch.startsWith("aarch64")) {
      detectedArch = "aarch64";
    } else if (arch.startsWith("ppc64")) {
      detectedArch = "ppc64";
    } else if (isAndroid()) {
      detectedArch = arch;
    } else {
      throw new IllegalStateException("Unsupported arch:" + arch);
    }

    return detectedOS + '-' + detectedArch;
  }

  /**
   * Check if we're running on Android.
   *
   * @return True if the property java.vendor equals The Android Project, false otherwise.
   */
  static boolean isAndroid() {
    return System.getProperty("java.vendor", "generic").equals("The Android Project");
  }

  /**
   * Marks the file for delete on exit.
   *
   * @param file The file to remove.
   */
  private static void cleanUp(File file) {
    if (!file.exists()) {
      return;
    }

    logger.log(Level.FINE, "Deleting " + file + " on exit");
    file.deleteOnExit();
  }

  /**
   * Load a shared library by name.
   *
   * <p>If the library path is not specified via a system property then it attempts to extract the
   * library from the classpath before loading it.
   *
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(String library, String resourcePath) throws IOException {
    if (isAndroid()) {
      // On Android, we simply use System.loadLibrary.
      // We only need to load the JNI library as it will load the GenAI native library and ORT
      // native library
      // via the library's dependencies.
      if (library == GENAI_JNI_LIBRARY_NAME) {
        logger.log(Level.INFO, "Loading native library '" + library + "'");
        System.loadLibrary(library);
      }

      return;
    }

    // 1) The user may skip loading of this library:
    String skip = System.getProperty("onnxruntime-genai.native." + library + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping load of native library '" + library + "'");
      return;
    }

    // Resolve the platform dependent library name.
    String libraryFileName = mapLibraryName(library);

    // 2) The user may explicitly specify the path to a directory containing all shared libraries:
    if (libraryDirPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryDirPathProperty);

      // TODO: Switch this to Path.of when the minimum Java version is 11.
      File libraryFile = Paths.get(libraryDirPathProperty, libraryFileName).toFile();
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }

      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 3) The user may explicitly specify the path to their shared library:
    String libraryPathProperty =
        System.getProperty("onnxruntime-genai.native." + library + ".path");
    if (libraryPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryPathProperty);
      File libraryFile = new File(libraryPathProperty);
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }

      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 4) try loading from resources or library path:
    Optional<File> extractedPath = extractFromResources(library, resourcePath);
    if (extractedPath.isPresent()) {
      // extracted library from resources
      System.load(extractedPath.get().getAbsolutePath());
      logger.log(Level.FINE, "Loaded native library '" + library + "' from resource path");
    } else {
      // failed to load library from resources, try to load it from the library path
      logger.log(
          Level.FINE, "Attempting to load native library '" + library + "' from library path");
      System.loadLibrary(library);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from library path");
    }
  }

  /**
   * Extracts the library from the classpath resources. returns optional.empty if it failed to
   * extract or couldn't be found.
   *
   * @param library The library name
   * @param baseResourcePath The base resource path
   * @return An optional containing the file if it is successfully extracted, or an empty optional
   *     if it failed to extract or couldn't be found.
   */
  private static Optional<File> extractFromResources(String library, String baseResourcePath) {
    String libraryFileName = mapLibraryName(library);
    String resourcePath = baseResourcePath + OS_ARCH_STR + '/' + libraryFileName;
    File tempFile = tempDirectory.resolve(libraryFileName).toFile();

    try (InputStream is = GenAI.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        // Not found in classpath resources
        return Optional.empty();
      } else {
        // Found in classpath resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to load native library '"
                + library
                + "' from resource path "
                + resourcePath
                + " copying to "
                + tempFile);

        byte[] buffer = new byte[4096];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }

        logger.log(Level.FINE, "Extracted native library '" + library + "' from resource path");
        return Optional.of(tempFile);
      }
    } catch (IOException e) {
      logger.log(
          Level.WARNING, "Failed to extract library '" + library + "' from the resources", e);
      return Optional.empty();
    } finally {
      cleanUp(tempFile);
    }
  }

  /**
   * Maps the library name into a platform dependent library filename. Converts macOS's "jnilib" to
   * "dylib" but otherwise is the same as System#mapLibraryName(String).
   *
   * @param library The library name
   * @return The library filename.
   */
  private static String mapLibraryName(String library) {
    return System.mapLibraryName(library).replace("jnilib", "dylib");
  }
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/GenAIException.java

```
src/java/src/main/java/ai/onnxruntime/genai/GenAIException.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** An exception which contains the error message and code produced by the native layer. */
public final class GenAIException extends Exception {
  GenAIException(String message) {
    super(message);
  }

  GenAIException(String message, Exception innerException) {
    super(message, innerException);
  }
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Generator.java

```
src/java/src/main/java/ai/onnxruntime/genai/Generator.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * The Generator class generates output using a model and generator parameters.
 *
 * <p>The expected usage is to loop until isDone returns false. Within the loop, call computeLogits
 * followed by generateNextToken.
 *
 * <p>The newly generated token can be retrieved with getLastTokenInSequence and decoded with
 * TokenizerStream.Decode.
 *
 * <p>After the generation process is done, GetSequence can be used to retrieve the complete
 * generated sequence if needed.
 */
public final class Generator implements AutoCloseable, Iterable<Integer> {
  private long nativeHandle = 0;

  /**
   * Constructs a Generator object with the given model and generator parameters.
   *
   * @param model The model.
   * @param generatorParams The generator parameters.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Generator(Model model, GeneratorParams generatorParams) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalArgumentException("model has been freed and is invalid");
    }

    if (generatorParams.nativeHandle() == 0) {
      throw new IllegalArgumentException("generatorParams has been freed and is invalid");
    }

    nativeHandle = createGenerator(model.nativeHandle(), generatorParams.nativeHandle());
  }

  /**
   * Returns an iterator over elements of type {@code Integer}. A new token is generated each time
   * next() is called, by calling computeLogits and generateNextToken.
   *
   * @return an Iterator.
   */
  @Override
  public java.util.Iterator<Integer> iterator() {
    return new Iterator();
  }

  /**
   * Checks if the generation process is done.
   *
   * @return true if the generation process is done, false otherwise.
   */
  public boolean isDone() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return isDone(nativeHandle);
  }

  /**
   * Add a Tensor as a model input.
   *
   * @param name Name of the model input the tensor will provide.
   * @param tensor Tensor to add.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setModelInput(String name, Tensor tensor) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (tensor.nativeHandle() == 0) {
      throw new IllegalArgumentException("tensor has been freed and is invalid");
    }

    setModelInput(nativeHandle, name, tensor.nativeHandle());
  }

  /**
   * Add a NamedTensors as a model input.
   *
   * @param namedTensors NamedTensors to add.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setInputs(NamedTensors namedTensors) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (namedTensors.nativeHandle() == 0) {
      throw new IllegalArgumentException("tensor has been freed and is invalid");
    }

    setInputs(nativeHandle, namedTensors.nativeHandle());
  }

  /**
   * Appends tokens to the generator.
   *
   * @param inputIDs The tokens to append.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void appendTokens(int[] inputIDs) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    appendTokens(nativeHandle, inputIDs);
  }

  /**
   * Appends token sequences to the generator.
   *
   * @param sequences The sequences to append.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void appendTokenSequences(Sequences sequences) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (sequences.nativeHandle() == 0) {
      throw new IllegalArgumentException("sequences has been freed and is invalid");
    }

    appendTokenSequences(nativeHandle, sequences.nativeHandle());
  }

  /**
   * Returns the token count in the generator.
   *
   * @return The number of tokens.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public long tokenCount() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenCount(nativeHandle);
  }

  /**
   * Rewinds the generator to the given length. This is useful when the user wants to rewind the
   * generator to a specific length and continue generating from that point.
   *
   * @param newLength The desired length in tokens after rewinding.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void rewindTo(long newLength) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    rewindTo(nativeHandle, newLength);
  }

  /**
   * Computes the logits from the model based on the input ids and the past state. The computed
   * logits are stored in the generator.
   *
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void generateNextToken() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    generateNextTokenNative(nativeHandle);
  }

  /**
   * Retrieves a sequence of token ids for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return An array of integers with the sequence token ids.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int[] getSequence(long sequenceIndex) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSequenceNative(nativeHandle, sequenceIndex);
  }

  /**
   * Retrieves the last token in the sequence for the specified sequence index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The last token in the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getLastTokenInSequence(long sequenceIndex) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSequenceLastToken(nativeHandle, sequenceIndex);
  }

  /**
   * Returns a copy of the model input identified by the given name as a Tensor.
   *
   * @param name The name of the input needed.
   * @return The tensor.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tensor getInput(String name) throws GenAIException {
    long tensorHandle = getInputNative(nativeHandle, name);
    return new Tensor(tensorHandle);
  }

  /**
   * Returns a copy of the model output identified by the given name as a Tensor.
   *
   * @param name The name of the output needed.
   * @return The tensor.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tensor getOutput(String name) throws GenAIException {
    long tensorHandle = getOutputNative(nativeHandle, name);
    return new Tensor(tensorHandle);
  }

  /**
   * Sets the adapter with the given adapter name as active.
   *
   * @param adapters The Adapters container.
   * @param adapterName The adapter name that was previously loaded.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setActiveAdapter(Adapters adapters, String adapterName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setActiveAdapter(nativeHandle, adapters.nativeHandle(), adapterName);
  }

  /** Closes the Generator and releases any associated resources. */
  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyGenerator(nativeHandle);
      nativeHandle = 0;
    }
  }

  /** The Iterator class for the Generator to simplify usage when streaming tokens. */
  private class Iterator implements java.util.Iterator<Integer> {
    @Override
    public boolean hasNext() {
      return !isDone();
    }

    @Override
    public Integer next() {
      try {
        generateNextToken();
        return getLastTokenInSequence(0);
      } catch (GenAIException e) {
        throw new RuntimeException(e);
      }
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGenerator(long modelHandle, long generatorParamsHandle)
      throws GenAIException;

  private native void destroyGenerator(long nativeHandle);

  private native boolean isDone(long nativeHandle);

  private native void setModelInput(long nativeHandle, String inputName, long tensorHandle)
      throws GenAIException;

  private native void setInputs(long nativeHandle, long namedTensorsHandle) throws GenAIException;

  private native void appendTokens(long nativeHandle, int[] tokens) throws GenAIException;

  private native void appendTokenSequences(long nativeHandle, long sequencesHandle)
      throws GenAIException;

  private native long tokenCount(long nativeHandle) throws GenAIException;

  private native void rewindTo(long nativeHandle, long newLength) throws GenAIException;

  private native void generateNextTokenNative(long nativeHandle) throws GenAIException;

  private native int[] getSequenceNative(long nativeHandle, long sequenceIndex)
      throws GenAIException;

  private native int getSequenceLastToken(long nativeHandle, long sequenceIndex)
      throws GenAIException;

  private native void setActiveAdapter(
      long nativeHandle, long adaptersNativeHandle, String adapterName) throws GenAIException;

  private native long getInputNative(long nativeHandle, String outputName) throws GenAIException;

  private native long getOutputNative(long nativeHandle, String outputName) throws GenAIException;
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/GeneratorParams.java

```
src/java/src/main/java/ai/onnxruntime/genai/GeneratorParams.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.nio.ByteBuffer;

/**
 * Represents the parameters used for generating sequences with a model. Set the prompt using
 * setInputs, and any other search options using setSearchOption.
 */
public final class GeneratorParams implements AutoCloseable {
  private long nativeHandle = 0;
  private ByteBuffer tokenIdsBuffer;

  /**
   * Creates a GeneratorParams from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public GeneratorParams(Model model) throws GenAIException {
    if (model.nativeHandle() == 0) {
      throw new IllegalStateException("model has been freed and is invalid");
    }

    nativeHandle = createGeneratorParams(model.nativeHandle());
  }

  /**
   * Set search option with double value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, double value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionNumber(nativeHandle, optionName, value);
  }

  /**
   * Set search option with boolean value.
   *
   * @param optionName The option name.
   * @param value The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void setSearchOption(String optionName, boolean value) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    setSearchOptionBool(nativeHandle, optionName, value);
  }

  /**
   * Get search option with numerical value.
   *
   * @param optionName The option name.
   * @return The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public double getSearchNumber(String optionName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSearchNumber(nativeHandle, optionName);
  }

  /**
   * Get search option with boolean value.
   *
   * @param optionName The option name.
   * @return The option value.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public boolean getSearchBool(String optionName) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSearchBool(nativeHandle, optionName);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyGeneratorParams(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createGeneratorParams(long modelHandle) throws GenAIException;

  private native void destroyGeneratorParams(long nativeHandle);

  private native void setSearchOptionNumber(long nativeHandle, String optionName, double value)
      throws GenAIException;

  private native void setSearchOptionBool(long nativeHandle, String optionName, boolean value)
      throws GenAIException;

  private native double getSearchNumber(long nativeHandle, String optionName) throws GenAIException;

  private native boolean getSearchBool(long nativeHandle, String optionName) throws GenAIException;
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Images.java

```
src/java/src/main/java/ai/onnxruntime/genai/Images.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** This class can load images from the given path and prepare them for processing. */
public class Images implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a Images instance.
   *
   * @param imagePath The image path.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Images(String imagePath) throws GenAIException {
    nativeHandle = loadImages(imagePath);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyImages(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long loadImages(String imagePath) throws GenAIException;

  private native void destroyImages(long imageshandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Model.java

```
src/java/src/main/java/ai/onnxruntime/genai/Model.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** An ORT GenAI model. */
public final class Model implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a Model from folder path.
   *
   * @param modelPath The path of the GenAI model.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Model(String modelPath) throws GenAIException {
    nativeHandle = createModel(modelPath);
  }

  /**
   * Construct a Model from the given Config.
   *
   * @param config The config to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Model(Config config) throws GenAIException {
    nativeHandle = createModelFromConfig(config.nativeHandle());
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyModel(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createModel(String modelPath) throws GenAIException;

  private native long createModelFromConfig(long configHandle) throws GenAIException;

  private native void destroyModel(long modelHandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/MultiModalProcessor.java

```
src/java/src/main/java/ai/onnxruntime/genai/MultiModalProcessor.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * The MultiModalProcessor class is responsible for converting text/images into a NamedTensors list
 * that can be fed into a Generator class instance.
 */
public class MultiModalProcessor implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a MultiModalProcessor for a given model.
   *
   * @param model The model to be used.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public MultiModalProcessor(Model model) throws GenAIException {
    assert (model.nativeHandle() != 0); // internal code should never pass an invalid model

    nativeHandle = createMultiModalProcessor(model.nativeHandle());
  }

  /**
   * Processes a string and image into a NamedTensor.
   *
   * @param prompt Text to encode as token ids.
   * @param images image input.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processImages(String prompt, Images images) throws GenAIException {
    long imagesHandle = (images == null) ? 0 : images.nativeHandle();
    long namedTensorsHandle = processorProcessImages(nativeHandle, prompt, imagesHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Processes strings and images into a NamedTensor.
   *
   * @param prompts Texts to encode as token ids.
   * @param images image inputs.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processImages(String[] prompts, Images images) throws GenAIException {
    long imagesHandle = (images == null) ? 0 : images.nativeHandle();
    long namedTensorsHandle = processorProcessImagesAndPrompts(nativeHandle, prompts, imagesHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Processes a string and audio into a NamedTensor.
   *
   * @param prompt Text to encode as token ids.
   * @param audios audio input.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processAudios(String prompt, Audios audios) throws GenAIException {
    long audiosHandle = (audios == null) ? 0 : audios.nativeHandle();
    long namedTensorsHandle = processorProcessAudios(nativeHandle, prompt, audiosHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Processes strings and audios into a NamedTensor.
   *
   * @param prompts Texts to encode as token ids.
   * @param audios audio inputs.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processAudios(String[] prompts, Audios audios) throws GenAIException {
    long audiosHandle = (audios == null) ? 0 : audios.nativeHandle();
    long namedTensorsHandle = processorProcessAudiosAndPrompts(nativeHandle, prompts, audiosHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Processes a string, images, and audios into a NamedTensor.
   *
   * @param prompt Text to encode as token ids.
   * @param images image input.
   * @param audios audio input.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processImagesAndAudios(String prompt, Images images, Audios audios)
      throws GenAIException {
    long imagesHandle = (images == null) ? 0 : images.nativeHandle();
    long audiosHandle = (audios == null) ? 0 : audios.nativeHandle();
    long namedTensorsHandle =
        processorProcessImagesAndAudios(nativeHandle, prompt, imagesHandle, audiosHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Processes strings, images, and audios into a NamedTensor.
   *
   * @param prompts Texts to encode as token ids.
   * @param images image inputs.
   * @param audios audio inputs.
   * @return NamedTensors object.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public NamedTensors processImagesAndAudios(String[] prompts, Images images, Audios audios)
      throws GenAIException {
    long imagesHandle = (images == null) ? 0 : images.nativeHandle();
    long audiosHandle = (audios == null) ? 0 : audios.nativeHandle();
    long namedTensorsHandle =
        processorProcessImagesAndAudiosAndPrompts(
            nativeHandle, prompts, imagesHandle, audiosHandle);

    return new NamedTensors(namedTensorsHandle);
  }

  /**
   * Decodes a sequence of token ids into text.
   *
   * @param sequence Collection of token ids to decode to text.
   * @return The text representation of the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int[] sequence) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return processorDecode(nativeHandle, sequence);
  }

  /**
   * Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
   * to provide each token as it is generated.
   *
   * @return The new TokenizerStream instance.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public TokenizerStream createStream() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
    return new TokenizerStream(createTokenizerStreamFromProcessor(nativeHandle));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyMultiModalProcessor(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createMultiModalProcessor(long modelHandle) throws GenAIException;

  private native void destroyMultiModalProcessor(long tokenizerHandle);

  private native long processorProcessImages(long processorHandle, String prompt, long imagesHandle)
      throws GenAIException;

  private native long processorProcessImagesAndPrompts(
      long processorHandle, String[] prompts, long imagesHandle) throws GenAIException;

  private native long processorProcessAudios(long processorHandle, String prompt, long audiosHandle)
      throws GenAIException;

  private native long processorProcessAudiosAndPrompts(
      long processorHandle, String[] prompts, long audiosHandle) throws GenAIException;

  private native long processorProcessImagesAndAudios(
      long processorHandle, String prompt, long imagesHandle, long audiosHandle)
      throws GenAIException;

  private native long processorProcessImagesAndAudiosAndPrompts(
      long processorHandle, String[] prompts, long imagesHandle, long audiosHandle)
      throws GenAIException;

  private native String processorDecode(long processorHandle, int[] sequence) throws GenAIException;

  private native long createTokenizerStreamFromProcessor(long processorHandle)
      throws GenAIException;
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/NamedTensors.java

```
src/java/src/main/java/ai/onnxruntime/genai/NamedTensors.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** This class is a list of tensors with names that match up with model input names. */
public class NamedTensors implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a NamedTensor from native handle.
   *
   * @param handle The native handle.
   */
  public NamedTensors(long handle) {
    nativeHandle = handle;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyNamedTensors(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native void destroyNamedTensors(long handle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Sequences.java

```
src/java/src/main/java/ai/onnxruntime/genai/Sequences.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Represents a collection of encoded prompts/responses. */
public final class Sequences implements AutoCloseable {
  private long nativeHandle;
  private long numSequences;

  Sequences(long sequencesHandle) {
    assert (sequencesHandle != 0); // internal usage should never pass an invalid handle

    nativeHandle = sequencesHandle;
    numSequences = getSequencesCount(sequencesHandle);
  }

  /**
   * Gets the number of sequences in the collection. This is equivalent to the batch size.
   *
   * @return The number of sequences.
   */
  public long numSequences() {
    return numSequences;
  }

  /**
   * Gets the sequence at the specified index.
   *
   * @param sequenceIndex The index of the sequence.
   * @return The sequence as an array of integers.
   */
  public int[] getSequence(long sequenceIndex) {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return getSequenceNative(nativeHandle, sequenceIndex);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroySequences(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long getSequencesCount(long sequencesHandle);

  private native int[] getSequenceNative(long sequencesHandle, long sequenceIndex);

  private native void destroySequences(long sequencesHandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/SimpleGenAI.java

```
src/java/src/main/java/ai/onnxruntime/genai/SimpleGenAI.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.function.Consumer;

/**
 * The `SimpleGenAI` class provides a simple usage example of the GenAI API. It works with a model
 * that generates text based on a prompt, processing a single prompt at a time.
 *
 * <p>Usage:
 *
 * <ul>
 *   <li>Create an instance of the class with the path to the model. The path should also contain
 *       the GenAI configuration files.
 *   <li>Call createGeneratorParams with the prompt text.
 *   <li>Set any other search options via the GeneratorParams object as needed using
 *       `setSearchOption`.
 *   <li>Call generate with the GeneratorParams object and an optional listener.
 * </ul>
 *
 * <p>The listener is used as a callback mechanism so that tokens can be used as they are generated.
 * It should be an instance of a type that implements the Consumer&lt;String&gt; interface.
 */
public class SimpleGenAI implements AutoCloseable {
  private Model model;
  private Tokenizer tokenizer;

  /**
   * Construct a SimpleGenAI instance from model path.
   *
   * @param modelPath The path to the GenAI model.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public SimpleGenAI(String modelPath) throws GenAIException {
    model = new Model(modelPath);
    tokenizer = new Tokenizer(model);
  }

  /**
   * Get the underlying model object.
   *
   * @return The model object.
   * @throws GenAIException on failure
   */
  public Model getModel() throws GenAIException {
    return model;
  }

  /**
   * Create the generator parameters and add the prompt text. The user can set other search options
   * via the GeneratorParams object prior to running `generate`.
   *
   * @return The generator parameters.
   * @throws GenAIException on failure
   */
  public GeneratorParams createGeneratorParams() throws GenAIException {
    return new GeneratorParams(model);
  }

  /**
   * Generate text based on the prompt and settings in GeneratorParams.
   *
   * <p>NOTE: This only handles a single sequence of input (i.e. a single prompt which equates to
   * batch size of 1)
   *
   * @param generatorParams The prompt and settings to run the model with.
   * @param prompt The prompt text to encode.
   * @param listener Optional callback for tokens to be provided as they are generated. NOTE: Token
   *     generation will be blocked until the listener's `accept` method returns. `listener` will be
   *     called within the token generation loop and these calls will be made sequentially, not
   *     concurrently.
   * @return The generated text.
   * @throws GenAIException on failure
   */
  public String generate(GeneratorParams generatorParams, String prompt, Consumer<String> listener)
      throws GenAIException {
    String result;
    try {
      int[] output_ids;

      if (listener != null) {
        try (TokenizerStream stream = tokenizer.createStream();
            Generator generator = new Generator(model, generatorParams)) {
          // iterate (which calls computeLogits, generateNextToken, getLastTokenInSequence and
          // isDone)
          generator.appendTokenSequences(tokenizer.encode(prompt));
          for (int token_id : generator) {
            // decode and call listener
            String token = stream.decode(token_id);
            listener.accept(token);
          }

          output_ids = generator.getSequence(0);
        } catch (GenAIException e) {
          throw new GenAIException("Token generation loop failed.", e);
        }
      } else {
        try (Generator generator = new Generator(model, generatorParams)) {
          generator.appendTokenSequences(tokenizer.encode(prompt));
          for (int token_id : generator) {
            // do nothing
          }
          output_ids = generator.getSequence(0);
        } catch (GenAIException e) {
          throw new GenAIException("Token generation loop failed.", e);
        }
      }

      result = tokenizer.decode(output_ids);
    } catch (GenAIException e) {
      throw new GenAIException("Failed to create Tokenizer.", e);
    }

    return result;
  }

  @Override
  public void close() {
    if (tokenizer != null) {
      tokenizer.close();
      tokenizer = null;
    }
    if (model != null) {
      model.close();
      model = null;
    }
  }
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Tensor.java

```
src/java/src/main/java/ai/onnxruntime/genai/Tensor.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Currently wraps an ORT Tensor. */
public final class Tensor implements AutoCloseable {
  private long nativeHandle = 0;
  private final ElementType elementType;
  private final long[] shape;

  // Buffer that owns the Tensor data.
  private ByteBuffer dataBuffer = null;

  // The values in this enum must match ONNX values
  // https://github.com/onnx/onnx/blob/159fa47b7c4d40e6d9740fcf14c36fff1d11ccd8/onnx/onnx.proto#L499-L544
  /** Element types that correspond to OnnxRuntime supported element types. */
  public enum ElementType {
    /** Undefined element type. */
    undefined,
    /** 32-bit IEEE 754 floating point. */
    float32,
    /** 8-bit unsigned integer. */
    uint8,
    /** 8-bit signed integer. */
    int8,
    /** 16-bit unsigned integer. */
    uint16,
    /** 16-bit signed integer. */
    int16,
    /** 32-bit signed integer. */
    int32,
    /** 64-bit signed integer. */
    int64,
    /** UTF-8 encoded string. */
    string,
    /** Boolean value (true or false). */
    bool,
    /** 16-bit IEEE 754 floating point (half precision). */
    float16,
    /** 64-bit IEEE 754 floating point (double precision). */
    float64,
    /** 32-bit unsigned integer. */
    uint32,
    /** 64-bit unsigned integer. */
    uint64,
  }

  /**
   * Constructs a Tensor with the given data, shape and element type.
   *
   * @param data The data for the Tensor. Must be a direct ByteBuffer with native byte order.
   * @param shape The shape of the Tensor.
   * @param elementType The type of elements in the Tensor.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tensor(ByteBuffer data, long[] shape, ElementType elementType) throws GenAIException {
    if (data == null || shape == null || elementType == ElementType.undefined) {
      throw new GenAIException(
          "Invalid input. data and shape must be provided, and elementType must not be undefined.");
    }

    // for now require the buffer to be direct.
    // we could support non-direct but need to do an allocate and copy here.
    if (!data.isDirect()) {
      throw new GenAIException(
          "Tensor data must be direct. Allocate with ByteBuffer.allocateDirect");
    }

    // for now, require native byte order as the bytes will be used directly.
    if (data.order() != ByteOrder.nativeOrder()) {
      throw new GenAIException("Tensor data must have native byte order.");
    }

    this.elementType = elementType;
    this.shape = shape;
    this.dataBuffer = data; // save a reference so the owning buffer will stay around.

    nativeHandle = createTensor(data, shape, elementType.ordinal());
  }

  /**
   * Construct a Tensor from native handle.
   *
   * @param handle The native tensor handle.
   */
  Tensor(long handle) {
    nativeHandle = handle;
    elementType = ElementType.values()[getTensorType(handle)];
    shape = getTensorShape(handle);
  }

  /**
   * Get the element type.
   *
   * @return The element type.
   */
  public ElementType getType() {
    return this.elementType;
  }

  /**
   * Get the tensor shape.
   *
   * @return The tensor type.
   */
  public long[] getShape() {
    return this.shape;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTensor(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createTensor(ByteBuffer data, long[] shape, int elementType)
      throws GenAIException;

  private native void destroyTensor(long tensorHandle);

  private native int getTensorType(long tensorHandle);

  private native long[] getTensorShape(long tensorHandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/Tokenizer.java

```
src/java/src/main/java/ai/onnxruntime/genai/Tokenizer.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** The Tokenizer class is responsible for converting between text and token ids. */
public class Tokenizer implements AutoCloseable {
  private long nativeHandle;

  /**
   * Creates a Tokenizer from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tokenizer(Model model) throws GenAIException {
    assert (model.nativeHandle() != 0); // internal code should never pass an invalid model

    nativeHandle = createTokenizer(model.nativeHandle());
  }

  /**
   * Encodes a string into a sequence of token ids.
   *
   * @param string Text to encode as token ids.
   * @return a Sequences object with a single sequence in it.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encode(String string) throws GenAIException {
    return encodeBatch(new String[] {string});
  }

  /**
   * Encodes an array of strings into a sequence of token ids for each input.
   *
   * @param strings Collection of strings to encode as token ids.
   * @return a Sequences object with one sequence per input string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encodeBatch(String[] strings) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    long sequencesHandle = tokenizerEncode(nativeHandle, strings);
    return new Sequences(sequencesHandle);
  }

  /**
   * Decodes a sequence of token ids into text.
   *
   * @param sequence Collection of token ids to decode to text.
   * @return The text representation of the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int[] sequence) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerDecode(nativeHandle, sequence);
  }

  /**
   * Decodes a batch of sequences of token ids into text.
   *
   * @param sequences A Sequences object with one or more sequences of token ids.
   * @return An array of strings with the text representation of each sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String[] decodeBatch(Sequences sequences) throws GenAIException {
    int numSequences = (int) sequences.numSequences();

    String[] result = new String[numSequences];
    for (int i = 0; i < numSequences; i++) {
      result[i] = decode(sequences.getSequence(i));
    }

    return result;
  }

  /**
   * Gets the beginning of sentence token ID.
   *
   * @return The BOS token ID.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getBosTokenId() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerGetBosTokenId(nativeHandle);
  }

  /**
   * Gets the padding token ID.
   *
   * @return The padding token ID.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getPadTokenId() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerGetPadTokenId(nativeHandle);
  }

  /**
   * Gets the end of sentence token IDs.
   *
   * @return An array of EOS token IDs.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int[] getEosTokenIds() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerGetEosTokenIds(nativeHandle);
  }

  /**
   * Converts a string to a token ID.
   *
   * @param str The string to convert to a token ID.
   * @return The token ID for the given string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int toTokenId(String str) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerToTokenId(nativeHandle, str);
  }

  /**
   * Applies a chat template to format messages.
   *
   * @param templateStr The template string to use.
   * @param messages The messages in JSON format.
   * @param tools The tools in JSON format (can be null).
   * @param addGenerationPrompt Whether to add generation prompt.
   * @return The formatted chat string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String applyChatTemplate(
      String templateStr, String messages, String tools, boolean addGenerationPrompt)
      throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerApplyChatTemplate(
        nativeHandle, templateStr, messages, tools, addGenerationPrompt);
  }

  /**
   * Updates tokenizer options.
   *
   * @param options Map of option keys to values.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void updateOptions(java.util.Map<String, String> options) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    if (options == null || options.isEmpty()) {
      return; // Nothing to update
    }

    String[] keys = options.keySet().toArray(new String[0]);
    String[] values = new String[keys.length];

    for (int i = 0; i < keys.length; i++) {
      values[i] = options.get(keys[i]);
    }

    tokenizerUpdateOptions(nativeHandle, keys, values);
  }

  /**
   * Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
   * to provide each token as it is generated.
   *
   * @return The new TokenizerStream instance.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public TokenizerStream createStream() throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return new TokenizerStream(createTokenizerStream(nativeHandle));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTokenizer(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createTokenizer(long modelHandle) throws GenAIException;

  private native void destroyTokenizer(long tokenizerHandle);

  private native long tokenizerEncode(long tokenizerHandle, String[] strings) throws GenAIException;

  private native String tokenizerDecode(long tokenizerHandle, int[] sequence) throws GenAIException;

  private native long createTokenizerStream(long tokenizerHandle) throws GenAIException;

  private native int tokenizerGetBosTokenId(long tokenizerHandle) throws GenAIException;

  private native int tokenizerGetPadTokenId(long tokenizerHandle) throws GenAIException;

  private native int[] tokenizerGetEosTokenIds(long tokenizerHandle) throws GenAIException;

  private native int tokenizerToTokenId(long tokenizerHandle, String str) throws GenAIException;

  private native String tokenizerApplyChatTemplate(
      long tokenizerHandle,
      String templateStr,
      String messages,
      String tools,
      boolean addGenerationPrompt)
      throws GenAIException;

  private native void tokenizerUpdateOptions(long tokenizerHandle, String[] keys, String[] values)
      throws GenAIException;
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/TokenizerStream.java

```
src/java/src/main/java/ai/onnxruntime/genai/TokenizerStream.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * A TokenizerStream is used to convert individual tokens when using Generator.generateNextToken.
 */
public class TokenizerStream implements AutoCloseable {

  private long nativeHandle = 0;

  /**
   * Construct a TokenizerStream.
   *
   * @param tokenizerStreamHandle The native handle.
   */
  TokenizerStream(long tokenizerStreamHandle) {
    assert (tokenizerStreamHandle != 0); // internal usage should never pass an invalid handle
    nativeHandle = tokenizerStreamHandle;
  }

  /**
   * Decode one token.
   *
   * @param token The token.
   * @return The decoded result.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int token) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerStreamDecode(nativeHandle, token);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTokenizerStream(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native String tokenizerStreamDecode(long tokenizerStreamHandle, int token)
      throws GenAIException;

  private native void destroyTokenizerStream(long tokenizerStreamHandle);
}
```


---

# FILE: src/java/src/main/java/ai/onnxruntime/genai/package-info.java

```
src/java/src/main/java/ai/onnxruntime/genai/package-info.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * A Java interface to the ONNX Runtime GenAI library.
 *
 * <p>There are two shared libraries required: <code>onnxruntime-genai</code> and <code>
 * onnxruntime-genai-jni
 * </code>. The loader is in ai.onnxruntime.genai.GenAI and the logic is in this order:
 *
 * <ol>
 *   <li>The user may signal to skip loading of a shared library using a property in the form <code>
 *       onnxruntime-genai.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means
 *       the user has decided to load the library by some other means.
 *   <li>The user may specify an explicit location of all native library files using a property in
 *       the form <code>onnxruntime-genai.native.path</code>. This uses {java.lang.System#load}.
 *   <li>The user may specify an explicit location of the shared library file using a property in
 *       the form <code>onnxruntime-genai.native.LIB_NAME.path</code>. This uses
 *       {java.lang.System#load}.
 *   <li>The shared library is autodiscovered:
 *       <ol>
 *         <li>If the shared library is present in the classpath resources, load using {
 *             java.lang.System#load} via a temporary file. Ideally, this should be the default use
 *             case when adding JAR's/dependencies containing the shared libraries to your
 *             classpath.
 *         <li>If the shared library is not present in the classpath resources, then load using
 *             {java.lang.System#loadLibrary}, which usually looks elsewhere on the filesystem for
 *             the library. The semantics and behavior of that method are system/JVM dependent.
 *             Typically, the <code>java.library.path</code> property is used to specify the
 *             location of native libraries.
 *       </ol>
 * </ol>
 *
 * For troubleshooting, all shared library loading events are reported to Java logging at the level
 * FINE.
 */
package ai.onnxruntime.genai;
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Adapters.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Adapters.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Adapters.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Adapters_createAdapters(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaAdapters* adapters = nullptr;
  if (ThrowIfError(env, OgaCreateAdapters(model, &adapters))) {
    return 0;
  }

  return reinterpret_cast<jlong>(adapters);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_destroyAdapters(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyAdapters(reinterpret_cast<OgaAdapters*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_loadAdapter(JNIEnv* env, jobject thiz, jlong native_handle,
                                               jstring adapter_file_path, jstring adapter_name) {
  CString file_path{env, adapter_file_path};
  CString name{env, adapter_name};
  ThrowIfError(env, OgaLoadAdapter(reinterpret_cast<OgaAdapters*>(native_handle), file_path, name));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_unloadAdapter(JNIEnv* env, jobject thiz, jlong native_handle, jstring adapter_name) {
  CString name{env, adapter_name};
  ThrowIfError(env, OgaUnloadAdapter(reinterpret_cast<OgaAdapters*>(native_handle), name));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Audios.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Audios.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Audios.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Audios
 * Method:    loadAudios
 * Signature: (J)V
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Audios_loadAudios(JNIEnv* env, jobject thiz, jstring audio_path) {
  CString path(env, audio_path);

  OgaAudios* audios = nullptr;
  if (ThrowIfError(env, OgaLoadAudio(path, &audios))) {
    return 0;
  }

  return reinterpret_cast<jlong>(audios);
}

/*
 * Class:     ai_onnxruntime_genai_Audios
 * Method:    destroyAudios
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Audios_destroyAudios(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyAudios(reinterpret_cast<OgaAudios*>(native_handle));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Config.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Config.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Config.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Config_createConfig(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaConfig* config = nullptr;
  if (ThrowIfError(env, OgaCreateConfig(path, &config))) {
    return 0;
  }

  return reinterpret_cast<jlong>(config);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_destroyConfig(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);
  OgaDestroyConfig(config);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_clearProviders(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);
  ThrowIfError(env, OgaConfigClearProviders(config));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_appendProvider(JNIEnv* env, jobject thiz, jlong native_handle, jstring provider_name) {
  CString c_provider_name{env, provider_name};
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);

  ThrowIfError(env, OgaConfigAppendProvider(config, c_provider_name));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_setProviderOption(JNIEnv* env, jobject thiz, jlong native_handle, jstring provider_name, jstring option_key, jstring option_value) {
  CString c_provider_name{env, provider_name};
  CString c_option_key{env, option_key};
  CString c_option_value{env, option_value};
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);

  ThrowIfError(env, OgaConfigSetProviderOption(config, c_provider_name, c_option_key, c_option_value));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_overlay(JNIEnv* env, jobject thiz, jlong native_handle, jstring json) {
  CString c_json{env, json};
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);

  ThrowIfError(env, OgaConfigOverlay(config, c_json));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_GenAI.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_GenAI.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_GenAI.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GenAI_shutdown(JNIEnv* env, jclass cls) {
  OgaShutdown();
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Generator.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Generator.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Generator.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Generator_createGenerator(JNIEnv* env, jobject thiz, jlong model_handle,
                                                    jlong generator_params_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
  OgaGenerator* generator = nullptr;
  if (ThrowIfError(env, OgaCreateGenerator(model, params, &generator))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_destroyGenerator(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_setModelInput(JNIEnv* env, jobject thiz, jlong native_handle,
                                                  jstring input_name, jlong tensor) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);
  CString name{env, input_name};
  OgaTensor* input_tensor = reinterpret_cast<OgaTensor*>(tensor);

  ThrowIfError(env, OgaGenerator_SetModelInput(generator, name, input_tensor));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_setInputs(JNIEnv* env, jobject thiz, jlong native_handle,
                                              jlong namedTensors) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);
  OgaNamedTensors* input_tensor = reinterpret_cast<OgaNamedTensors*>(namedTensors);

  ThrowIfError(env, OgaGenerator_SetInputs(generator, input_tensor));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_appendTokenSequences(JNIEnv* env, jobject thiz, jlong native_handle,
                                                         jlong sequences_handle) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);

  ThrowIfError(env, OgaGenerator_AppendTokenSequences(generator, sequences));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_appendTokens(JNIEnv* env, jobject thiz, jlong native_handle, jintArray token_ids) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);

  jint* tokens = env->GetIntArrayElements(token_ids, nullptr);
  jsize num_tokens = env->GetArrayLength(token_ids);

  ThrowIfError(env, OgaGenerator_AppendTokens(generator, reinterpret_cast<int32_t*>(tokens), num_tokens));

  env->ReleaseIntArrayElements(token_ids, tokens, JNI_ABORT);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Generator_tokenCount(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);
  size_t count = OgaGenerator_TokenCount(generator);
  return static_cast<jlong>(count);
}

JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_genai_Generator_isDone(JNIEnv* env, jobject thiz, jlong native_handle) {
  return OgaGenerator_IsDone(reinterpret_cast<OgaGenerator*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_rewindTo(JNIEnv* env, jobject thiz, jlong native_handle, jlong length) {
  ThrowIfError(env, OgaGenerator_RewindTo(reinterpret_cast<OgaGenerator*>(native_handle), length));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_generateNextTokenNative(JNIEnv* env, jobject thiz, jlong native_handle) {
  ThrowIfError(env, OgaGenerator_GenerateNextToken(reinterpret_cast<OgaGenerator*>(native_handle)));
}

JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_genai_Generator_getSequenceNative(JNIEnv* env, jobject thiz, jlong generator, jlong index) {
  const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);

  size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, index);
  const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, index);

  if (num_tokens == 0) {
    ThrowException(env, "OgaGenerator_GetSequenceCount returned 0 tokens.");
    return nullptr;
  }

  // as there's no 'destroy' function in GenAI C API for the tokens we assume the OgaGenerator owns the memory.
  // copy the tokens so there's no potential for Java code to write to it (values should be treated as const)
  // or attempt to access the memory after the OgaGenerator is destroyed.
  jintArray java_int_array = env->NewIntArray(static_cast<jsize>(num_tokens));
  // jint is `long` on Windows and `int` on linux. 32-bit but requires reinterpret_cast.
  env->SetIntArrayRegion(java_int_array, 0, static_cast<jsize>(num_tokens), reinterpret_cast<const jint*>(tokens));

  return java_int_array;
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Generator_getSequenceLastToken(JNIEnv* env, jobject thiz, jlong generator, jlong index) {
  const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);

  size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, index);
  const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, index);

  if (num_tokens == 0) {
    ThrowException(env, "OgaGenerator_GetSequenceCount returned 0 tokens.");
    return -1;
  }

  return jint(tokens[num_tokens - 1]);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_setActiveAdapter(JNIEnv* env, jobject thiz, jlong native_handle,
                                                     jlong adapters_native_handle, jstring adapter_name) {
  CString name{env, adapter_name};
  ThrowIfError(env, OgaSetActiveAdapter(reinterpret_cast<OgaGenerator*>(native_handle),
                                        reinterpret_cast<OgaAdapters*>(adapters_native_handle),
                                        name));
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Generator_getInputNative(JNIEnv* env, jobject thiz, jlong native_handle,
                                                   jstring input_name) {
  OgaTensor* tensor = nullptr;
  CString name{env, input_name};
  if (ThrowIfError(env, OgaGenerator_GetInput(reinterpret_cast<OgaGenerator*>(native_handle), name, &tensor))) {
    return 0;
  }
  return reinterpret_cast<jlong>(tensor);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Generator_getOutputNative(JNIEnv* env, jobject thiz, jlong native_handle,
                                                    jstring output_name) {
  OgaTensor* tensor = nullptr;
  CString name{env, output_name};
  if (ThrowIfError(env, OgaGenerator_GetOutput(reinterpret_cast<OgaGenerator*>(native_handle), name, &tensor))) {
    return 0;
  }
  return reinterpret_cast<jlong>(tensor);
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_GeneratorParams.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_GeneratorParams.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_GeneratorParams.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_createGeneratorParams(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaGeneratorParams* generator_params = nullptr;
  if (ThrowIfError(env, OgaCreateGeneratorParams(model, &generator_params))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_destroyGeneratorParams(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  OgaDestroyGeneratorParams(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionNumber(JNIEnv* env, jobject thiz, jlong native_handle,
                                                                jstring option_name, jdouble value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchNumber(generator_params, name, value));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionBool(JNIEnv* env, jobject thiz, jlong native_handle,
                                                              jstring option_name, jboolean value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchBool(generator_params, name, value));
}

JNIEXPORT jdouble JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_getSearchNumber(JNIEnv* env, jobject thiz, jlong native_handle, jstring option_name) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};
  double value;

  ThrowIfError(env, OgaGeneratorParamsGetSearchNumber(generator_params, name, &value));
  return static_cast<jdouble>(value);
}

JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_getSearchBool(JNIEnv* env, jobject thiz, jlong native_handle, jstring option_name) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};
  bool value;

  ThrowIfError(env, OgaGeneratorParamsGetSearchBool(generator_params, name, &value));
  return static_cast<jboolean>(value);
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Images.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Images.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Images.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Images
 * Method:    loadImages
 * Signature: (J)V
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Images_loadImages(JNIEnv* env, jobject thiz, jstring image_path) {
  CString path(env, image_path);

  OgaImages* images = nullptr;
  if (ThrowIfError(env, OgaLoadImage(path, &images))) {
    return 0;
  }

  return reinterpret_cast<jlong>(images);
}

/*
 * Class:     ai_onnxruntime_genai_Images
 * Method:    destroyImages
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Images_destroyImages(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyImages(reinterpret_cast<OgaImages*>(native_handle));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Model.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Model.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Model.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModel(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaModel* model = nullptr;
  if (ThrowIfError(env, OgaCreateModel(path, &model))) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModelFromConfig(JNIEnv* env, jobject thiz, jlong config_handle) {
  const OgaConfig* config = reinterpret_cast<const OgaConfig*>(config_handle);

  OgaModel* model = nullptr;
  if (ThrowIfError(env, OgaCreateModelFromConfig(config, &model))) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Model_destroyModel(JNIEnv* env, jobject thiz, jlong model_handle) {
  OgaModel* model = reinterpret_cast<OgaModel*>(model_handle);
  OgaDestroyModel(model);
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_MultiModalProcessor.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_MultiModalProcessor.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_MultiModalProcessor.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_createMultiModalProcessor(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaMultiModalProcessor* processor = nullptr;

  if (ThrowIfError(env, OgaCreateMultiModalProcessor(model, &processor))) {
    return 0;
  }

  return reinterpret_cast<jlong>(processor);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_destroyMultiModalProcessor(JNIEnv* env, jobject thiz, jlong processor_handle) {
  OgaMultiModalProcessor* processor = reinterpret_cast<OgaMultiModalProcessor*>(processor_handle);
  OgaDestroyMultiModalProcessor(processor);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImages(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                     jstring prompt, jlong images_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);

  OgaNamedTensors* named_tensors = nullptr;
  bool error = ThrowIfError(env, OgaProcessorProcessImages(processor, prompt_str, images, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                               jobjectArray prompts, jlong images_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndPrompts(processor, strs, images, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessAudios(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                     jstring prompt, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  OgaNamedTensors* named_tensors = nullptr;
  bool error = ThrowIfError(env, OgaProcessorProcessAudios(processor, prompt_str, audios, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessAudiosAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                               jobjectArray prompts, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessAudiosAndPrompts(processor, strs, audios, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndAudios(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                              jstring prompt, jlong images_handle, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  OgaNamedTensors* named_tensors = nullptr;
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndAudios(processor, prompt_str, images, audios, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndAudiosAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                                        jobjectArray prompts, jlong images_handle, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndAudiosAndPrompts(processor, strs, images, audios, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorDecode(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                              jintArray sequence) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);
  auto num_tokens = env->GetArrayLength(sequence);
  jint* jtokens = env->GetIntArrayElements(sequence, nullptr);
  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types
  const char* decoded_text = nullptr;

  bool error = ThrowIfError(env, OgaProcessorDecode(processor, tokens, num_tokens, &decoded_text));
  env->ReleaseIntArrayElements(sequence, jtokens, JNI_ABORT);

  if (error) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_createTokenizerStreamFromProcessor(JNIEnv* env, jobject thiz, jlong processor_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);
  OgaTokenizerStream* tokenizer_stream = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizerStreamFromProcessor(processor, &tokenizer_stream))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer_stream);
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_NamedTensors.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_NamedTensors.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_NamedTensors.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_NamedTensors
 * Method:    destroyNamedTensors
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_NamedTensors_destroyNamedTensors(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyNamedTensors(reinterpret_cast<OgaNamedTensors*>(native_handle));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Sequences.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Sequences.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Sequences.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Sequences_destroySequences(JNIEnv* env, jobject thiz, jlong sequences_handle) {
  OgaSequences* sequences = reinterpret_cast<OgaSequences*>(sequences_handle);
  OgaDestroySequences(sequences);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Sequences_getSequencesCount(JNIEnv* env, jobject thiz, jlong sequences_handle) {
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);
  size_t num_sequences = OgaSequencesCount(sequences);
  return static_cast<jlong>(num_sequences);
}

JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_genai_Sequences_getSequenceNative(JNIEnv* env, jobject thiz, jlong sequences_handle,
                                                      jlong sequence_index) {
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);

  size_t num_tokens = OgaSequencesGetSequenceCount(sequences, (size_t)sequence_index);
  const int32_t* tokens = OgaSequencesGetSequenceData(sequences, (size_t)sequence_index);

  // as there's no 'destroy' function in GenAI C API for the tokens we assume OgaSequences owns the memory.
  // copy the tokens so there's no potential for Java code to write to it (values should be treated as const),
  // or attempt to access the memory after the OgaSequences is destroyed.
  // note: jint is `long` on Windows and `int` on linux. both are 32-bit but require reinterpret_cast.
  jintArray java_int_array = env->NewIntArray(static_cast<jsize>(num_tokens));
  env->SetIntArrayRegion(java_int_array, 0, static_cast<jsize>(num_tokens), reinterpret_cast<const jint*>(tokens));

  return java_int_array;
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Tensor.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Tensor.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Tensor.h"

#include "ort_genai_c.h"
#include "utils.h"

#include <vector>

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Tensor
 * Method:    createTensor
 * Signature: (Ljava/nio/ByteBuffer;[JI)J
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tensor_createTensor(JNIEnv* env, jobject thiz, jobject tensor_data,
                                              jlongArray shape_dims_in, jint element_type_in) {
  void* data = env->GetDirectBufferAddress(tensor_data);
  const int64_t* shape_dims = reinterpret_cast<int64_t*>(env->GetLongArrayElements(shape_dims_in, /*isCopy*/ 0));
  size_t shape_dims_count = env->GetArrayLength(shape_dims_in);
  OgaElementType element_type = static_cast<OgaElementType>(element_type_in);
  OgaTensor* tensor = nullptr;

  if (ThrowIfError(env, OgaCreateTensorFromBuffer(data, shape_dims, shape_dims_count, element_type, &tensor))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tensor);
}

/*
 * Class:     ai_onnxruntime_genai_Tensor
 * Method:    destroyTensor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tensor_destroyTensor(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyTensor(reinterpret_cast<OgaTensor*>(native_handle));
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tensor_getTensorType(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaElementType type;
  if (ThrowIfError(env, OgaTensorGetType(reinterpret_cast<OgaTensor*>(native_handle), &type))) {
    return 0;
  }
  return static_cast<int>(type);
}

JNIEXPORT jlongArray JNICALL
Java_ai_onnxruntime_genai_Tensor_getTensorShape(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaTensor* tensor = reinterpret_cast<OgaTensor*>(native_handle);
  size_t size;
  if (ThrowIfError(env, OgaTensorGetShapeRank(tensor, &size))) {
    return nullptr;
  }
  std::vector<int64_t> shape(size);
  if (ThrowIfError(env, OgaTensorGetShape(tensor, shape.data(), shape.size()))) {
    return nullptr;
  }

  jlongArray result;
  result = env->NewLongArray(static_cast<jsize>(size));
  static_assert(sizeof(jlong) == sizeof(int64_t));
  env->SetLongArrayRegion(result, 0, static_cast<jsize>(size), reinterpret_cast<const jlong*>(shape.data()));
  return result;
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_Tokenizer.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_Tokenizer.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Tokenizer.h"

#include "ort_genai_c.h"
#include "utils.h"
#include <vector>
#include <optional>

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_createTokenizer(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaTokenizer* tokenizer = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizer(model, &tokenizer))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tokenizer_destroyTokenizer(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  OgaTokenizer* tokenizer = reinterpret_cast<OgaTokenizer*>(tokenizer_handle);
  OgaDestroyTokenizer(tokenizer);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerEncode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                    jobjectArray strings) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_strings = env->GetArrayLength(strings);

  OgaSequences* sequences = nullptr;
  if (ThrowIfError(env, OgaCreateSequences(&sequences))) {
    return 0;
  }

  for (int i = 0; i < num_strings; i++) {
    jstring string = static_cast<jstring>(env->GetObjectArrayElement(strings, i));
    CString c_string{env, string};
    if (ThrowIfError(env, OgaTokenizerEncode(tokenizer, c_string, sequences))) {
      OgaDestroySequences(sequences);
      return 0;
    }
  }

  return reinterpret_cast<jlong>(sequences);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerDecode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                    jintArray sequence) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_tokens = env->GetArrayLength(sequence);
  jint* jtokens = env->GetIntArrayElements(sequence, nullptr);
  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types
  const char* decoded_text = nullptr;

  bool error = ThrowIfError(env, OgaTokenizerDecode(tokenizer, tokens, num_tokens, &decoded_text));
  env->ReleaseIntArrayElements(sequence, jtokens, JNI_ABORT);

  if (error) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_createTokenizerStream(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  OgaTokenizerStream* tokenizer_stream = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizerStream(tokenizer, &tokenizer_stream))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer_stream);
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetBosTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerGetBosTokenId(tokenizer, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetPadTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerGetPadTokenId(tokenizer, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetEosTokenIds(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  const int32_t* eos_token_ids = nullptr;
  size_t token_count = 0;

  if (ThrowIfError(env, OgaTokenizerGetEosTokenIds(tokenizer, &eos_token_ids, &token_count))) {
    return nullptr;
  }

  // Create Java int array
  jintArray result = env->NewIntArray(static_cast<jsize>(token_count));
  if (result == nullptr) {
    return nullptr;  // OutOfMemoryError thrown
  }

  // Copy the token IDs to the Java array
  env->SetIntArrayRegion(result, 0, static_cast<jsize>(token_count),
                         reinterpret_cast<const jint*>(eos_token_ids));

  return result;
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerToTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle, jstring str) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  CString c_string{env, str};
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerToTokenId(tokenizer, c_string, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerApplyChatTemplate(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                               jstring template_str, jstring messages, jstring tools, jboolean add_generation_prompt) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  CString c_template_str{env, template_str};
  CString c_messages{env, messages};

  std::optional<CString> c_tools;
  const char* c_tools_ptr = nullptr;
  if (tools != nullptr) {
    c_tools = CString{env, tools};
    c_tools_ptr = *c_tools;
  }

  const char* result = nullptr;

  if (ThrowIfError(env, OgaTokenizerApplyChatTemplate(tokenizer, c_template_str, c_messages, c_tools_ptr, add_generation_prompt, &result))) {
    return nullptr;
  }

  jstring jresult = env->NewStringUTF(result);
  OgaDestroyString(result);

  return jresult;
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerUpdateOptions(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                           jobjectArray keys, jobjectArray values) {
  OgaTokenizer* tokenizer = reinterpret_cast<OgaTokenizer*>(tokenizer_handle);
  auto num_options = env->GetArrayLength(keys);

  if (num_options != env->GetArrayLength(values)) {
    Helpers::ThrowException(env, "Keys and values arrays must have the same length");
    return;
  }

  // Convert Java string arrays to C string arrays
  std::vector<CString> c_keys;
  std::vector<CString> c_values;
  std::vector<const char*> c_keys_ptr;
  std::vector<const char*> c_values_ptr;
  c_keys.reserve(num_options);
  c_values.reserve(num_options);
  c_keys_ptr.reserve(num_options);
  c_values_ptr.reserve(num_options);

  for (int i = 0; i < num_options; i++) {
    jstring key = static_cast<jstring>(env->GetObjectArrayElement(keys, i));
    jstring value = static_cast<jstring>(env->GetObjectArrayElement(values, i));

    c_keys.emplace_back(env, key);
    c_values.emplace_back(env, value);

    c_keys_ptr.push_back(c_keys.back());
    c_values_ptr.push_back(c_values.back());
  }

  ThrowIfError(env, OgaUpdateTokenizerOptions(tokenizer, c_keys_ptr.data(), c_values_ptr.data(), num_options));
}
```


---

# FILE: src/java/src/main/native/ai_onnxruntime_genai_TokenizerStream.cpp

```
src/java/src/main/native/ai_onnxruntime_genai_TokenizerStream.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_TokenizerStream.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_TokenizerStream_tokenizerStreamDecode(JNIEnv* env, jobject thiz,
                                                                jlong tokenizer_stream_handle, jint token) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  const char* decoded_text = nullptr;

  // The const char* returned in decoded_text is the result of calling c_str on a std::string in the tokenizer cache.
  // The std::string is owned by the tokenizer cache.
  // Due to that, it is invalid to call `OgaDestroyString(decoded_text)`, and doing so will result in a crash.
  if (ThrowIfError(env, OgaTokenizerStreamDecode(tokenizer_stream, token, &decoded_text))) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  return result;
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_TokenizerStream_destroyTokenizerStream(JNIEnv* env, jobject thiz,
                                                                 jlong tokenizer_stream_handle) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  OgaDestroyTokenizerStream(tokenizer_stream);
}
```


---

# FILE: src/java/src/main/native/utils.cpp

```
src/java/src/main/native/utils.cpp
```

```cpp
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "utils.h"

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  // To silence unused-parameter error.
  // This function must exist according to the JNI spec, but the arguments aren't necessary for the library
  // to request a specific version.
  (void)vm;
  (void)reserved;
  // Requesting 1.6 to support Android. Will need to be bumped to a later version to call interface default methods
  // from native code, or to access other new Java features.
  return JNI_VERSION_1_6;
}

namespace {
void ThrowExceptionImpl(JNIEnv* env, const char* error_message) {
  static const char* className = "ai/onnxruntime/genai/GenAIException";
  env->ThrowNew(env->FindClass(className), error_message);
}
}  // namespace

namespace Helpers {
void ThrowException(JNIEnv* env, const char* message) {
  ThrowExceptionImpl(env, message);
}

bool ThrowIfError(JNIEnv* env, OgaResult* result) {
  bool error = result != nullptr;

  if (error) {
    ThrowExceptionImpl(env, OgaResultGetError(result));
    OgaDestroyResult(result);
  }

  return error;
}
}  // namespace Helpers
```


---

# FILE: src/java/src/main/native/utils.h

```
src/java/src/main/native/utils.h
```

```h
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "ort_genai_c.h"

#ifdef _WIN32
#define strdup _strdup
#endif

#ifdef __cplusplus
extern "C" {
#endif

jint JNI_OnLoad(JavaVM* vm, void* reserved);

#ifdef __cplusplus
}
#endif

namespace Helpers {
void ThrowException(JNIEnv* env, const char* message);

/// @brief Throw a GenAIException if the result is an error.
/// @param env JNI environment
/// @param result Result from GenAI C API call
/// @return True if there was an error. JNI code should generally return immediately if this is true.
bool ThrowIfError(JNIEnv* env, OgaResult* result);

// handle conversion/release of jstring to const char*
struct CString {
  CString(JNIEnv* env, jstring str)
      : cstr{env->GetStringUTFChars(str, /* isCopy */ nullptr)}, env_{env}, str_{str} {
  }

  const char* cstr;

  operator const char*() const { return cstr; }

  ~CString() {
    env_->ReleaseStringUTFChars(str_, cstr);
  }

 private:
  JNIEnv* env_;
  jstring str_;
};
}  // namespace Helpers
```


---

# FILE: src/java/src/test/android/README.md

```
src/java/src/test/android/README.md
```

```md
# Android Test Application for ONNX Runtime GenAI

This directory contains a simple android application for testing the ONNX Runtime GenaI AAR package.

### Test Android Application Overview

This android application is mainly aimed for testing:

- Model used: test/models/hf-internal-testing/tiny-random-gpt2-fp32
- Main test file: An android instrumentation test under `app\src\androidtest\java\ai.onnxruntime.genai.example.javavalidator\SimpleTest.kt`
- The main dependency of this application is `onnxruntime-genai` aar package under `app\libs`.
- The onnxruntime dependency is provided by the latest released onnxruntime-android package.
- The MainActivity of this application is set to be empty.

### Requirements

- JDK version 11 or later is required.
- The [Gradle](https://gradle.org/) build system is required for building the APKs used to run [android instrumentation tests](https://source.android.com/compatibility/tests/development/instrumentation). Version 7.5 or newer is required.
  The Gradle wrapper at `java/gradlew[.bat]` may be used.

### Building

Build for Android with the additional  `--build_java` and `--android_run_emulator` options.

e.g.
`./build --android --android_home D:\Android --android_ndk_path D:\Android\ndk\26.3.11579264\ --android_abi x86_64 --ort_home 'path to unzipped onnxruntime-android.aar from https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/<version>' --build_java --android_run_emulator`

Please note that you must set the `--android_abi` value to match the local system architecture, as the Android instrumentation test is run on an Android emulator on the local system.

See ../../AndroidBuild.md for more information on building for Android.

#### Build Output

The build will generate two apks which is required to run the test application in `$YOUR_BUILD_DIR/src/java/androidtest/app/build/outputs/apk`:

* `androidTest/debug/app-debug-androidTest.apk`
* `debug/app-debug.apk`

After running the build script, the two apks will be installed on `ort_genai_android` emulator and it will automatically run the test application in an adb shell.
```


---

# FILE: src/java/src/test/android/app/build.gradle

```
src/java/src/test/android/app/build.gradle
```

```gradle
plugins {
	id 'com.android.application'
	id 'kotlin-android'
}

def minSdkVer = System.properties.get("minSdkVer")?:27

android {
	compileSdkVersion 32

	defaultConfig {
		applicationId "ai.onnxruntime.genai.example.javavalidator"
		minSdkVersion minSdkVer
		targetSdkVersion 32
		versionCode 1
		versionName "1.0"

		testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
	}

	buildTypes {
		release {
			minifyEnabled false
			proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
		}
	}
	compileOptions {
		sourceCompatibility JavaVersion.VERSION_1_8
		targetCompatibility JavaVersion.VERSION_1_8
	}
	kotlinOptions {
		jvmTarget = '1.8'
	}
	namespace 'ai.onnxruntime.genai.example.javavalidator'
}

dependencies {
	implementation "org.jetbrains.kotlin:kotlin-stdlib-jdk8:$kotlin_version"
	implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
	implementation 'androidx.core:core-ktx:1.3.2'
	implementation 'androidx.appcompat:appcompat:1.2.0'
	implementation 'com.google.android.material:material:1.3.0'
	implementation 'androidx.constraintlayout:constraintlayout:2.0.4'
	implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
	implementation(name: "onnxruntime-genai", ext: "aar")

	testImplementation 'junit:junit:4.+'
	androidTestImplementation 'androidx.test.ext:junit:1.1.3'
	androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'
	androidTestImplementation 'androidx.test:runner:1.4.0'
	androidTestImplementation 'androidx.test:rules:1.4.0'
}
```


---

# FILE: src/java/src/test/android/app/proguard-rules.pro

```
src/java/src/test/android/app/proguard-rules.pro
```

```pro
# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile
```


---

# FILE: src/java/src/test/android/app/src/androidTest/java/ai/onnxruntime_genai/example/javavalidator/SimpleTest.kt

```
src/java/src/test/android/app/src/androidTest/java/ai/onnxruntime_genai/example/javavalidator/SimpleTest.kt
```

```kt
package ai.onnxruntime.genai.example.javavalidator

import ai.onnxruntime.genai.*
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.*
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*


private const val TAG = "ORTGenAIAndroidTest"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    val activityTestRule = ActivityScenarioRule(MainActivity::class.java)

    @Before
    fun start() {
        Log.i(TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @Throws(IOException::class)
    private fun copyModelFromAssets(): String {
        // NOTE: We have to read from the app's assets (app/src/main/assets) and write the the app's filesDir.
        // The unit test's context cannot be used. you'll get mysterious errors like File.mkdirs() will return false and
        // assertManager.open() throws even if the filename is valid if you try and use it.
        // Test context is InstrumentationRegistry.getInstrumentation().targetContext.
        val context = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext
        val assetManager = context.assets
        val files: Array<String>?
        try {
            files = context.assets.list("model")
        } catch (e: IOException) {
            Log.e("copyModelFromAssets", "Failed to find `model` folder in app assets.", e)
            throw e
        }

        val filesDir = context.filesDir
        if (!filesDir.exists()) {
            throw IOException("Files directory is not valid: " + filesDir.absolutePath)
        }

        val modelTargetPath = filesDir.absolutePath + File.separator + "model"
        val modelTargetDir = File(modelTargetPath)
        if (!modelTargetDir.exists() and !modelTargetDir.mkdirs()) {
            throw IOException("Target directory could not be created: " + modelTargetDir.absolutePath)
        }

        // the model data is expected to be large, so use a decent sized buffer
        val buffer = ByteArray(64*1024)

        for (filename in files!!) {
            try {
                val outFile = File(modelTargetPath + File.separator + filename)
                if (!outFile.exists()) {
                    outFile.createNewFile()
                }

                val srcStream = assetManager.open("model/$filename")
                val dstStream = FileOutputStream(outFile)
                var bytesRead: Int
                while (srcStream.read(buffer).also { bytesRead = it } != -1) {
                    dstStream.write(buffer, 0, bytesRead)
                }

                srcStream.close()
                dstStream.flush()
                dstStream.close()
            } catch (e: IOException) {
                Log.e("copyModelFromAssets", "Failed to copy file from assets/model: $filename", e)
            }
        }

        return modelTargetPath
    }

    @Test
    fun runBasicTest() {
        // We have to copy the model directory from the app assets to the app's files directory so it can be accessed
        // normally by the C++ GenAI code. when it's in the apk it is compressed.
        val newModelPath = copyModelFromAssets()

        // the test model requires manual input as the token ids have to be < 1000 but the configured tokenizer
        // has a larger vocab size and the input ids it generates are not valid.
        val model = Model(newModelPath)
        val params = GeneratorParams(model)

        val sequenceLength = 4
        val batchSize = 2
        val tokenIds: IntArray = intArrayOf(0, 0, 0, 52,
                                            0, 0, 195, 731)

        val maxLength = 10
        params.setSearchOption("max_length", maxLength.toDouble())
        params.setSearchOption("batch_size", batchSize.toDouble())

        val generator = Generator(model, params)
        generator.appendTokens(tokenIds)
        while(!generator.isDone()) {
            generator.generateNextToken()
        }

        val expectedOutput =
            intArrayOf(
                0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
                0, 0, 195, 731, 731, 114, 114, 114, 114, 114
            )

        for (i in 0 until batchSize) {
            val outputIds: IntArray = generator.getSequence(i.toLong())
            for (j in 0 until maxLength) {
                Assert.assertEquals(outputIds[j], expectedOutput[i * maxLength + j])
            }
        }

        Log.i("runBasicTest", "GenAI output matched expected data.")
    }
}
```


---

# FILE: src/java/src/test/android/app/src/main/AndroidManifest.xml

```
src/java/src/test/android/app/src/main/AndroidManifest.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.JavaValidator">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
```


---

# FILE: src/java/src/test/android/app/src/main/java/ai/onnxruntime_genai/example/javavalidator/MainActivity.kt

```
src/java/src/test/android/app/src/main/java/ai/onnxruntime_genai/example/javavalidator/MainActivity.kt
```

```kt
package ai.onnxruntime.genai.example.javavalidator

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/*Empty activity app mainly used for testing*/
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }
}
```


---

# FILE: src/java/src/test/android/app/src/main/res/drawable/ic_launcher_background.xml

```
src/java/src/test/android/app/src/main/res/drawable/ic_launcher_background.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp"
    android:height="108dp"
    android:viewportWidth="108"
    android:viewportHeight="108">
    <path
        android:fillColor="#3DDC84"
        android:pathData="M0,0h108v108h-108z" />
    <path
        android:fillColor="#00000000"
        android:pathData="M9,0L9,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,0L19,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M29,0L29,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M39,0L39,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M49,0L49,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M59,0L59,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M69,0L69,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M79,0L79,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M89,0L89,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M99,0L99,108"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,9L108,9"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,19L108,19"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,29L108,29"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,39L108,39"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,49L108,49"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,59L108,59"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,69L108,69"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,79L108,79"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,89L108,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M0,99L108,99"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,29L89,29"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,39L89,39"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,49L89,49"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,59L89,59"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,69L89,69"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M19,79L89,79"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M29,19L29,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M39,19L39,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M49,19L49,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M59,19L59,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M69,19L69,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
    <path
        android:fillColor="#00000000"
        android:pathData="M79,19L79,89"
        android:strokeWidth="0.8"
        android:strokeColor="#33FFFFFF" />
</vector>
```


---

# FILE: src/java/src/test/android/app/src/main/res/drawable-v24/ic_launcher_foreground.xml

```
src/java/src/test/android/app/src/main/res/drawable-v24/ic_launcher_foreground.xml
```

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:aapt="http://schemas.android.com/aapt"
    android:width="108dp"
    android:height="108dp"
    android:viewportWidth="108"
    android:viewportHeight="108">
    <path android:pathData="M31,63.928c0,0 6.4,-11 12.1,-13.1c7.2,-2.6 26,-1.4 26,-1.4l38.1,38.1L107,108.928l-32,-1L31,63.928z">
        <aapt:attr name="android:fillColor">
            <gradient
                android:endX="85.84757"
                android:endY="92.4963"
                android:startX="42.9492"
                android:startY="49.59793"
                android:type="linear">
                <item
                    android:color="#44000000"
                    android:offset="0.0" />
                <item
                    android:color="#00000000"
                    android:offset="1.0" />
            </gradient>
        </aapt:attr>
    </path>
    <path
        android:fillColor="#FFFFFF"
        android:fillType="nonZero"
        android:pathData="M65.3,45.828l3.8,-6.6c0.2,-0.4 0.1,-0.9 -0.3,-1.1c-0.4,-0.2 -0.9,-0.1 -1.1,0.3l-3.9,6.7c-6.3,-2.8 -13.4,-2.8 -19.7,0l-3.9,-6.7c-0.2,-0.4 -0.7,-0.5 -1.1,-0.3C38.8,38.328 38.7,38.828 38.9,39.228l3.8,6.6C36.2,49.428 31.7,56.028 31,63.928h46C76.3,56.028 71.8,49.428 65.3,45.828zM43.4,57.328c-0.8,0 -1.5,-0.5 -1.8,-1.2c-0.3,-0.7 -0.1,-1.5 0.4,-2.1c0.5,-0.5 1.4,-0.7 2.1,-0.4c0.7,0.3 1.2,1 1.2,1.8C45.3,56.528 44.5,57.328 43.4,57.328L43.4,57.328zM64.6,57.328c-0.8,0 -1.5,-0.5 -1.8,-1.2s-0.1,-1.5 0.4,-2.1c0.5,-0.5 1.4,-0.7 2.1,-0.4c0.7,0.3 1.2,1 1.2,1.8C66.5,56.528 65.6,57.328 64.6,57.328L64.6,57.328z"
        android:strokeWidth="1"
        android:strokeColor="#00000000" />
</vector>
```


---

# FILE: src/java/src/test/android/app/src/main/res/layout/activity_main.xml

```
src/java/src/test/android/app/src/main/res/layout/activity_main.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintLeft_toLeftOf="parent"
        app:layout_constraintRight_toRightOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>
```


---

# FILE: src/java/src/test/android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml

```
src/java/src/test/android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/ic_launcher_background" />
    <foreground android:drawable="@drawable/ic_launcher_foreground" />
</adaptive-icon>
```


---

# FILE: src/java/src/test/android/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml

```
src/java/src/test/android/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/ic_launcher_background" />
    <foreground android:drawable="@drawable/ic_launcher_foreground" />
</adaptive-icon>
```


---

# FILE: src/java/src/test/android/app/src/main/res/values/colors.xml

```
src/java/src/test/android/app/src/main/res/values/colors.xml
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="purple_200">#FFBB86FC</color>
    <color name="purple_500">#FF6200EE</color>
    <color name="purple_700">#FF3700B3</color>
    <color name="teal_200">#FF03DAC5</color>
    <color name="teal_700">#FF018786</color>
    <color name="black">#FF000000</color>
    <color name="white">#FFFFFFFF</color>
</resources>
```


---

# FILE: src/java/src/test/android/app/src/main/res/values/strings.xml

```
src/java/src/test/android/app/src/main/res/values/strings.xml
```

```xml
<resources>
    <string name="app_name">JavaValidator</string>
</resources>
```


---

# FILE: src/java/src/test/android/app/src/main/res/values/themes.xml

```
src/java/src/test/android/app/src/main/res/values/themes.xml
```

```xml
<resources xmlns:tools="http://schemas.android.com/tools">
    <!-- Base application theme. -->
    <style name="Theme.JavaValidator" parent="Theme.MaterialComponents.DayNight.DarkActionBar">
        <!-- Primary brand color. -->
        <item name="colorPrimary">@color/purple_500</item>
        <item name="colorPrimaryVariant">@color/purple_700</item>
        <item name="colorOnPrimary">@color/white</item>
        <!-- Secondary brand color. -->
        <item name="colorSecondary">@color/teal_200</item>
        <item name="colorSecondaryVariant">@color/teal_700</item>
        <item name="colorOnSecondary">@color/black</item>
        <!-- Status bar color. -->
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimaryVariant</item>
        <!-- Customize your theme here. -->
    </style>
</resources>
```


---

# FILE: src/java/src/test/android/app/src/main/res/values-night/themes.xml

```
src/java/src/test/android/app/src/main/res/values-night/themes.xml
```

```xml
<resources xmlns:tools="http://schemas.android.com/tools">
    <!-- Base application theme. -->
    <style name="Theme.JavaValidator" parent="Theme.MaterialComponents.DayNight.DarkActionBar">
        <!-- Primary brand color. -->
        <item name="colorPrimary">@color/purple_200</item>
        <item name="colorPrimaryVariant">@color/purple_700</item>
        <item name="colorOnPrimary">@color/black</item>
        <!-- Secondary brand color. -->
        <item name="colorSecondary">@color/teal_200</item>
        <item name="colorSecondaryVariant">@color/teal_200</item>
        <item name="colorOnSecondary">@color/black</item>
        <!-- Status bar color. -->
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimaryVariant</item>
        <!-- Customize your theme here. -->
    </style>
</resources>
```


---

# FILE: src/java/src/test/android/build.gradle

```
src/java/src/test/android/build.gradle
```

```gradle
// Top-level build file where you can add configuration options common to all sub-projects/modules.
buildscript {
	ext.kotlin_version = '1.6.21'

	repositories {
		google()
		mavenCentral()
	}
	dependencies {
		classpath 'com.android.tools.build:gradle:7.4.2'
		classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
		// NOTE: Do not place your application dependencies here; they belong
		// in the individual module build.gradle files
	}
}

allprojects {
	repositories {
		google()
		mavenCentral()
		flatDir{dirs 'libs'}
	}
}

task clean(type: Delete) {
	delete rootProject.buildDir
}
```


---

# FILE: src/java/src/test/android/gradle.properties

```
src/java/src/test/android/gradle.properties
```

```properties
# Project-wide Gradle settings.
# IDE (e.g. Android Studio) users:
# Gradle settings configured through the IDE *will override*
# any settings specified in this file.
# For more details on how to configure your build environment visit
# http://www.gradle.org/docs/current/userguide/build_environment.html
# Specifies the JVM arguments used for the daemon process.
# The setting is particularly useful for tweaking memory settings.
org.gradle.jvmargs=-Xmx4096m -Dfile.encoding=UTF-8
# When configured, Gradle will run in incubating parallel mode.
# This option should only be used with decoupled projects. More details, visit
# http://www.gradle.org/docs/current/userguide/multi_project_builds.html#sec:decoupled_projects
# org.gradle.parallel=true
# AndroidX package structure to make it clearer which packages are bundled with the
# Android operating system, and which are packaged with your app"s APK
# https://developer.android.com/topic/libraries/support-library/androidx-rn
android.useAndroidX=true
# Automatically convert third-party libraries to use AndroidX
android.enableJetifier=true
# Kotlin code style for this project: "official" or "obsolete":
kotlin.code.style=official
```


---

# FILE: src/java/src/test/android/settings.gradle

```
src/java/src/test/android/settings.gradle
```

```gradle
include ':app'
rootProject.name = "JavaValidator"
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/GenAITestExecutionListener.java

```
src/java/src/test/java/ai/onnxruntime/genai/GenAITestExecutionListener.java
```

```java
package ai.onnxruntime.genai;

import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestPlan;

public class GenAITestExecutionListener implements TestExecutionListener {
  public void testPlanExecutionFinished(TestPlan testPlan) {
    GenAI.shutdown();
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/GenerationTest.java

```
src/java/src/test/java/ai/onnxruntime/genai/GenerationTest.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Consumer;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

// Test the overall generation.
// Uses SimpleGenAI with phi-2 (if available) for text -> text generation.
// Uses the HF test model with pre-defined input tokens for token -> token generation
//
// This indirectly tests the majority of the bindings. Any gaps are covered in the class specific
// tests.
public class GenerationTest {
  private static final Logger logger = Logger.getLogger(GenerationTest.class.getName());

  // phi-2 can be used in full end-to-end testing but needs to be manually downloaded.
  // it's also used this way in the C# unit tests.
  private static final String phi2ModelPath() {
    return TestUtils.getTestModelPath("phi-2/int4/cpu");
  }

  @SuppressWarnings("unused") // Used in EnabledIf
  private static boolean havePhi2() {
    return phi2ModelPath() != null;
  }

  @SuppressWarnings("unused") // Used in EnabledIf
  private static boolean haveAdapters() {
    return TestUtils.testAdapterTestModelPath() != null;
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUsageNoListener() throws GenAIException {
    try (SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      params.setSearchOption("max_length", 20);
      String result =
          generator.generate(params, TestUtils.applyPhi2ChatTemplate("What's 6 times 7?"), null);
      logger.info("Result: " + result);
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUsageWithListener() throws GenAIException {
    try (SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      params.setSearchOption("max_length", 20);
      Consumer<String> listener = token -> logger.info("onTokenGenerate: " + token);
      String result =
          generator.generate(
              params, TestUtils.applyPhi2ChatTemplate("What's 6 times 7?"), listener);

      logger.info("Result: " + result);
    }
  }

  @Test
  @EnabledIf("haveAdapters")
  public void testUsageWithAdapters() throws GenAIException {
    try (Model model = new Model(TestUtils.testAdapterTestModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      String[] prompts = {
        TestUtils.applyPhi2ChatTemplate("def is_prime(n):"),
        TestUtils.applyPhi2ChatTemplate("def compute_gcd(x, y):"),
        TestUtils.applyPhi2ChatTemplate("def binary_search(arr, x):"),
      };

      try (Sequences sequences = tokenizer.encodeBatch(prompts);
          GeneratorParams params = new GeneratorParams(model)) {
        params.setSearchOption("max_length", 200);
        params.setSearchOption("batch_size", prompts.length);

        long[] outputShape;

        try (Generator generator = new Generator(model, params); ) {
          generator.appendTokenSequences(sequences);
          while (!generator.isDone()) {
            generator.generateNextToken();
          }

          try (Tensor logits = generator.getOutput("logits")) {
            outputShape = logits.getShape();
            assertEquals(logits.getType(), Tensor.ElementType.float32);
          }
        }

        try (Adapters adapters = new Adapters(model);
            Generator generator = new Generator(model, params); ) {
          generator.appendTokenSequences(sequences);
          adapters.loadAdapter(TestUtils.testAdapterTestAdaptersPath(), "adapters_a_and_b");
          generator.setActiveAdapter(adapters, "adapters_a_and_b");
          while (!generator.isDone()) {
            generator.generateNextToken();
          }
          try (Tensor logits = generator.getOutput("logits")) {
            assertEquals(logits.getType(), Tensor.ElementType.float32);
            assertArrayEquals(outputShape, logits.getShape());
          }
        }
      }
    }
  }

  @Test
  public void testWithInputIds() throws GenAIException {
    // test using the HF model. input id values must be < 1000 so we use manually created input.
    // Input/expected output copied from the C# unit tests
    try (Config config = new Config(TestUtils.tinyGpt2ModelPath());
        Model model = new Model(config);
        GeneratorParams params = new GeneratorParams(model); ) {
      int batchSize = 2;
      int sequenceLength = 4;
      int maxLength = 10;
      int[] inputIDs =
          new int[] {
            0, 0, 0, 52,
            0, 0, 195, 731
          };

      params.setSearchOption("max_length", maxLength);
      params.setSearchOption("batch_size", batchSize);

      int[] expectedOutput =
          new int[] {
            0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
            0, 0, 195, 731, 731, 114, 114, 114, 114, 114
          };

      try (Generator generator = new Generator(model, params); ) {
        generator.appendTokens(inputIDs);

        assertEquals(params.getSearchNumber("max_length"), maxLength);
        assertEquals(params.getSearchBool("early_stopping"), true);
        assertEquals(generator.tokenCount(), 4);

        while (!generator.isDone()) {
          generator.generateNextToken();
        }

        for (int i = 0; i < batchSize; i++) {
          int[] outputIds = generator.getSequence(i);
          for (int j = 0; j < maxLength; j++) {
            assertEquals(outputIds[j], expectedOutput[i * maxLength + j]);
          }
        }
        assertEquals(generator.tokenCount(), generator.getSequence(0).length);
      }
    }
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/GeneratorParamsTest.java

```
src/java/src/test/java/ai/onnxruntime/genai/GeneratorParamsTest.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class GeneratorParamsTest {
  @Test
  public void testValidSearchOption() throws GenAIException {
    // test setting a valid search option
    try (SimpleGenAI generator = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      params.setSearchOption("early_stopping", true); // boolean
      params.setSearchOption("max_length", 20); // number
    }
  }

  @Test
  public void testInvalidSearchOption() throws GenAIException {
    // test setting an invalid search option throws a GenAIException
    try (SimpleGenAI generator = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      assertThrows(GenAIException.class, () -> params.setSearchOption("invalid", true));
    }
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/MultiModalProcessorTest.java

```
src/java/src/test/java/ai/onnxruntime/genai/MultiModalProcessorTest.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.logging.Logger;
import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class MultiModalProcessorTest {
  private static final Logger logger = Logger.getLogger(MultiModalProcessorTest.class.getName());

  @Test
  public void testBatchEncodeDecode() throws GenAIException {
    try (Model model = new Model(TestUtils.testVisionModelPath());
        MultiModalProcessor multiModalProcessor = new MultiModalProcessor(model);
        TokenizerStream stream = multiModalProcessor.createStream();
        GeneratorParams generatorParams = new GeneratorParams(model)) {
      String inputs =
          new String(
              "<|user|>\n<|image_1|>\n Can you convert the table to markdown format?\n<|end|>\n<|assistant|>\n");
      try (Images image =
              new Images(TestUtils.getFilePathFromDisk(TestUtils.getTestImagePath("sheet.png")));
          NamedTensors processed = multiModalProcessor.processImages(inputs, image); ) {
        assertNotNull(processed);
      }
    }
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/TensorTest.java

```
src/java/src/test/java/ai/onnxruntime/genai/TensorTest.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.junit.jupiter.api.Test;

public class TensorTest {

  @Test
  public void testAddTensorInput() throws GenAIException {
    // test setting an invalid search option throws a GenAIException
    try (SimpleGenAI genAI = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = genAI.createGeneratorParams(); ) {
      long[] shape = {2, 2};
      Tensor.ElementType elementType = Tensor.ElementType.float32;
      ByteBuffer data = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());

      FloatBuffer floatBuffer = data.asFloatBuffer();
      floatBuffer.put(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
      try (Tensor tensor = new Tensor(data, shape, elementType)) {
        // no error on setting.
        // assuming there's an error on execution if an invalid input has been provided so the user
        // is aware of the issue
        Model model = genAI.getModel();
        Generator generator = new Generator(model, params);
        generator.setModelInput("unknown_value", tensor);
      }
    }
  }

  @Test
  public void testInvalidParams() throws GenAIException {
    // ByteBuffer that is not directly allocated
    long[] shape = {2, 2};
    Tensor.ElementType elementType = Tensor.ElementType.float32;
    ByteBuffer indirect_data = ByteBuffer.allocate(4 * Float.BYTES);
    assertThrows(GenAIException.class, () -> new Tensor(indirect_data, shape, elementType));

    // missing data
    assertThrows(GenAIException.class, () -> new Tensor(null, shape, elementType));

    ByteBuffer data = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());

    // missing shape
    assertThrows(GenAIException.class, () -> new Tensor(data, null, elementType));

    // undefined data type
    assertThrows(GenAIException.class, () -> new Tensor(data, shape, Tensor.ElementType.undefined));
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/TestUtils.java

```
src/java/src/test/java/ai/onnxruntime/genai/TestUtils.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.io.File;
import java.net.URL;
import java.util.logging.Logger;

public class TestUtils {
  private static final Logger logger = Logger.getLogger(TestUtils.class.getName());

  public static final String testAdapterTestModelPath() {
    return getFilePathFromDisk(getTestModelPath("adapters"));
  }

  public static final String testAdapterTestAdaptersPath() {
    return getFilePathFromDisk(getTestModelPath("adapters/adapters.onnx_adapter"));
  }

  public static final String tinyGpt2ModelPath() {
    return getFilePathFromDisk(getTestModelPath("hf-internal-testing/tiny-random-gpt2-fp32"));
  }

  public static final String phi2ModelPath() {
    return getFilePathFromDisk(getTestModelPath("phi-2/int4/cpu"));
  }

  public static final String testVisionModelPath() {
    return getFilePathFromDisk(getTestModelPath("phi3-v"));
  }

  public static final String getTestModelPath(String relativeResourcePath) {
    return getFilePathFromDisk(getRepoRoot() + "test/models/" + relativeResourcePath);
  }

  public static final String getTestImagePath(String relativeResourcePath) {
    return getFilePathFromDisk(getRepoRoot() + "test/images/" + relativeResourcePath);
  }

  public static final String getRepoRoot() {
    String classDirFileUrl = SimpleGenAI.class.getResource("").getFile();
    String repoRoot = classDirFileUrl.substring(0, classDirFileUrl.lastIndexOf("src/java/build"));
    return repoRoot;
  }

  public static final boolean setLocalNativeLibraryPath() {
    // set to <build output dir>/src/java/native-jni/ai/onnxruntime/genai/native/win-x64,
    // adjusting for your build output location and platform as needed
    String nativeJniBuildOutput =
        "build/Windows/Debug/src/java/native-jni/ai/onnxruntime/genai/native/win-x64";
    File fullPath = new File(getRepoRoot() + nativeJniBuildOutput);
    if (!fullPath.exists()) {
      logger.warning("Local native-jni build output not found at: " + fullPath.getPath());
      return false;
    }

    System.setProperty("onnxruntime-genai.native.path", fullPath.getPath());
    return true;
  }

  public static final String getFilePathFromResource(String path) {
    // get the resources directory from one of the classes
    URL url = TestUtils.class.getResource(path);
    if (url == null) {
      logger.warning("Model not found at " + path);
      return null;
    }

    File f = new File(url.getFile());
    return f.getPath();
  }

  public static final String getFilePathFromDisk(String path) {
    if (path == null) {
      logger.warning("Path provided is null");
      return null;
    }
    File f = new File(path);
    if (!f.exists()) {
      logger.warning("Model not found at " + path);
      return null;
    }

    return f.getPath();
  }

  public static final String applyPhi2ChatTemplate(String question) {
    return "User: " + question + "Assistant:";
  }

  public static final String applyPhi3ChatTemplate(String question) {
    return "<|user|>" + question + "<|end|><|assistant|>";
  }
}
```


---

# FILE: src/java/src/test/java/ai/onnxruntime/genai/TokenizerTest.java

```
src/java/src/test/java/ai/onnxruntime/genai/TokenizerTest.java
```

```java
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class TokenizerTest {
  @SuppressWarnings("unused") // Used in EnabledIf
  private static boolean havePhi2() {
    return TestUtils.phi2ModelPath() != null;
  }

  @Test
  public void testBatchEncodeDecode() throws GenAIException {
    try (Model model = new Model(TestUtils.tinyGpt2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      String[] inputs = new String[] {"This is a test", "This is another test"};
      try (Sequences encoded = tokenizer.encodeBatch(inputs)) {
        String[] decoded = tokenizer.decodeBatch(encoded);

        assertEquals(inputs.length, decoded.length);
        for (int i = 0; i < inputs.length; i++) {
          assert inputs[i].equals(decoded[i]);
        }
      }
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testGetBosTokenId() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int bosTokenId = tokenizer.getBosTokenId();
      assertTrue(bosTokenId == 50256, "BOS token ID should be 50256");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testGetEosTokenIds() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int[] eosTokenIds = tokenizer.getEosTokenIds();
      assertNotNull(eosTokenIds, "EOS token IDs should not be null");
      assertTrue(eosTokenIds.length == 1, "Should have exactly one EOS token sequence");

      if (eosTokenIds.length > 0) {
        assertTrue(eosTokenIds[0] == 50256, "First EOS token ID should be 50256");
      }
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testGetPadTokenId() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int padTokenId = tokenizer.getPadTokenId();
      assertTrue(padTokenId == 50256, "Pad token ID should be 50256");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testApplyChatTemplate() throws GenAIException {
    // We load the phi-2 model just to get a tokenizer (phi-2 does not have a chat template)
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      // Testing phi-4-mini chat template
      String messagesJson =
          "[\n"
              + "  {\n"
              + "    \"role\": \"system\",\n"
              + "    \"content\": \"You are a helpful assistant.\",\n"
              + "    \"tools\": \"[{\\\"name\\\": \\\"calculate_sum\\\", \\\"description\\\": \\\"Calculate the sum of two numbers.\\\", \\\"parameters\\\": {\\\"a\\\": {\\\"type\\\": \\\"int\\\"}, \\\"b\\\": {\\\"type\\\": \\\"int\\\"}}}]\"\n"
              + "  },\n"
              + "  {\n"
              + "    \"role\": \"user\",\n"
              + "    \"content\": \"How do I add two numbers?\"\n"
              + "  },\n"
              + "  {\n"
              + "    \"role\": \"assistant\",\n"
              + "    \"content\": \"You can add numbers by using the '+' operator.\"\n"
              + "  }\n"
              + "]";

      String chatTemplate =
          "{% for message in messages %}{% if message['role']"
              + " == 'system' and 'tools' in message and message['tools']"
              + " is not none %}{{ '<|' + message['role'] + '|>' + message['content']"
              + " + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}"
              + "{% else %}{{ '<|' + message['role'] + '|>' + message['content']"
              + " + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}"
              + "{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}";

      // From HuggingFace Python output for 'microsoft/Phi-4-mini-instruct'
      String expectedOutput =
          "<|system|>You are a helpful assistant.<|tool|>[{\"name\": \"calculate_sum\", \"description\": \"Calculate the sum of two numbers.\", \"parameters\": {\"a\": {\"type\": \"int\"}, \"b\": {\"type\": \"int\"}}}]<|/tool|><|end|><|user|>"
              + "How do I add two numbers?<|end|><|assistant|>You can add numbers by using the \"+\" operator.<|end|><|assistant|>";

      String result = tokenizer.applyChatTemplate(chatTemplate, messagesJson, null, true);
      assertEquals(expectedOutput, result, "Chat template output should match expected result");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUpdateOptionsWithMap() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      // Test with valid options map
      Map<String, String> options = new HashMap<>();
      options.put("add_special_tokens", "true");
      options.put("skip_special_tokens", "true");

      // This should not throw an exception
      tokenizer.updateOptions(options);
    }
  }
}
```


---

# FILE: src/java/src/test/resources/META-INF/services/org.junit.platform.launcher.TestExecutionListener

```
src/java/src/test/resources/META-INF/services/org.junit.platform.launcher.TestExecutionListener
```

```TestExecutionListener
ai.onnxruntime.genai.GenAITestExecutionListener
```


---

# FILE: src/java/windows-unittests.cmake

```
src/java/windows-unittests.cmake
```

```cmake
# This is a windows only file so we can run gradle tests via ctest

# Are these needed?
FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${JAVA_NATIVE_LIB_DIR} PACKAGE_LIB_DIR_NATIVE_PATH)

execute_process(COMMAND cmd /C ${GRADLE_NATIVE_PATH} 
                    --console=plain 
                    cmakeCheck 
                    spotlessApply
                    -DcmakeBuildDir=${BINDIR_NATIVE_PATH} 
                    -DnativeLibDir=${PACKAGE_LIB_DIR_NATIVE_PATH} 
                    -Dorg.gradle.daemon=false
                WORKING_DIRECTORY ${JAVA_SRC_ROOT}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
    message(FATAL_ERROR "Java Unitests failed")
endif()
```


---

# FILE: .github/workflows/build-oga-android-qnn.yml

```
.github/workflows/build-oga-android-qnn.yml
```

```yml
name: Build OGA Android AAR (QNN)

on:
  workflow_dispatch:

env:
  ORT_COMMIT: 312f5249437a4f923a6b288902bd802532f7b159
  NDK_VERSION: r27c
  ANDROID_API: 24
  QNN_HOME: /opt/qnn-sdk

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Checkout OGA source
        uses: actions/checkout@v4
        with:
          submodules: recursive
          path: onnxruntime-genai

      - name: Checkout ORT source (for QNN build)
        uses: actions/checkout@v4
        with:
          repository: microsoft/onnxruntime
          ref: ${{ env.ORT_COMMIT }}
          submodules: recursive
          path: onnxruntime
          fetch-depth: 1

      - name: Setup Java 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip -m install requests

      - name: Install Linux dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build jq ccache zip

      - name: Cache QNN SDK
        id: cache-qnn
        uses: actions/cache/restore@v4
        with:
          path: /opt/qnn-sdk
          key: qnn-sdk-2.46.0.260424-extracted

      - name: Download and extract QNN SDK (if cache missed)
        if: steps.cache-qnn.outputs.cache-hit != 'true'
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p /opt/qnn-sdk
          gh release download qnn-sdk-2.46.0.260424 \
            --repo catfewd/temp-hexagon-compiler \
            --pattern "qnn-sdk-2.46.0.260424-*.tar.gz" \
            --dir /tmp \
            --clobber
          tar xzf /tmp/qnn-sdk-*.tar.gz -C /opt/qnn-sdk
          cp -r /opt/qnn-sdk/qnn-sdk-min/* /opt/qnn-sdk/
          rm -rf /opt/qnn-sdk/qnn-sdk-min

      - name: Install Android NDK ${{ env.NDK_VERSION }}
        run: |
          curl -sL -o ndk.zip "https://dl.google.com/android/repository/android-ndk-r27c-linux.zip"
          unzip -q ndk.zip
          echo "ANDROID_NDK_HOME=$(pwd)/android-ndk-r27c" >> $GITHUB_ENV
          echo "ANDROID_NDK_ROOT=$(pwd)/android-ndk-r27c" >> $GITHUB_ENV
          echo "ANDROID_SDK_ROOT=/usr/local/lib/android/sdk" >> $GITHUB_ENV

      - name: Setup Rust toolchain
        run: |
          curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
          source $HOME/.cargo/env
          rustup install 1.86.0
          rustup override set 1.86.0
          rustup component add rust-src
          rustup target add x86_64-linux-android

      - name: Setup ccache (Reuse ORT cache)
        uses: actions/cache@v4
        with:
          path: $HOME/.ccache
          key: ccache-${{ runner.os }}-${{ env.ORT_COMMIT }}-qnn
          restore-keys: |
            ccache-${{ runner.os }}-${{ env.ORT_COMMIT }}-qnn

      - name: Build ORT with QNN (Fast due to cache)
        run: |
          cd onnxruntime
          export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CCACHE_CPP2=yes
          ./build.sh \
            --build_shared_lib \
            --android \
            --config MinSizeRel \
            --parallel 4 \
            --use_qnn static_lib \
            --qnn_home ${{ env.QNN_HOME }} \
            --android_ndk_path ${{ env.ANDROID_NDK_HOME }} \
            --android_sdk_path ${{ env.ANDROID_SDK_ROOT }} \
            --android_abi arm64-v8a \
            --android_api ${{ env.ANDROID_API }} \
            --cmake_generator Ninja \
            --build_dir build/Android \
            --skip_tests \
            --update --build

      - name: Build OGA (Fixed Arguments)
        run: |
          cd onnxruntime-genai
          python build.py \
            --build_dir build_android \
            --config RelWithDebInfo \
            --android \
            --android_abi arm64-v8a \
            --android_api ${{ env.ANDROID_API }} \
            --android_ndk_path $ANDROID_NDK_HOME \
            --android_home $ANDROID_SDK_ROOT \
            --build_java \
            --skip_wheel \
            --update \
            --build

      - name: Patch OGA AAR with QNN ORT
        run: |
          # Find the built AAR
          AAR=$(find onnxruntime-genai/build_android/RelWithDebInfo/src/java/ai/onnxruntime/genai/build/outputs/aar -name "*.aar" | head -n 1)
          echo "Found AAR: $AAR"
          
          # Extract the AAR
          mkdir -p aar_extract
          unzip -o "$AAR" -d aar_extract
          
          # Replace libonnxruntime.so with our QNN build
          QNN_SO=onnxruntime/build/Android/MinSizeRel/libonnxruntime.so
          cp -v "$QNN_SO" aar_extract/jni/arm64-v8a/libonnxruntime.so
          
          # Repackage the AAR
          cd aar_extract
          zip -r "../oga-qnn-patched.aar" .
          cd ..
          
          # Move to a known location for upload
          mv oga-qnn-patched.aar "$(dirname "$AAR")/oga-android-qnn.aar"

      - name: Upload Patched AAR
        uses: actions/upload-artifact@v4
        with:
          name: oga-android-qnn-aar
          path: onnxruntime-genai/build_android/RelWithDebInfo/src/java/ai/onnxruntime/genai/build/outputs/aar/oga-android-qnn.aar
          if-no-files-found: error
```


---

