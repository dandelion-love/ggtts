// The Android build: the binding as an AAR (ggtts) and the demo app that uses it (demo). scripts/android_build.sh runs
// it after building the native library, and checks what it made; see android/README.md.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ggtts-android"
include(":ggtts", ":demo")
