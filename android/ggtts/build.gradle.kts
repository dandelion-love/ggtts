// The binding as an Android library (AAR): Ggtts.kt and the native library built for arm64-v8a.
//
// The native library is not built here. scripts/android_build.sh builds libggtts_ffi.so with cargo-ndk (and copies
// the libraries it needs that a device lacks, libc++_shared.so) into a jniLibs directory, then passes it as the
// Gradle property ggtts.jniLibs. Without it the AAR would hold no native code, so this refuses to build.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val jniLibsDir: String = providers.gradleProperty("ggtts.jniLibs").orNull
    ?: throw GradleException(
        "set -Pggtts.jniLibs=<dir holding arm64-v8a/libggtts_ffi.so>; scripts/android_build.sh does this",
    )

android {
    namespace = "ggtts"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += "arm64-v8a" }
    }

    sourceSets["main"].jniLibs.srcDir(jniLibsDir)
    // The host-JVM test in src/test is a program run by scripts/kotlin_binding_test.sh, not a Gradle unit test.
    sourceSets["test"].kotlin.setSrcDirs(emptyList<String>())

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
