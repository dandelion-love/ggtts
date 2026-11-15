// The demo app: one screen that loads the model files the user picks and speaks one line with either engine, through
// the binding in :ggtts (built as an AAR). No model ships with it; see android/README.md. Its screen is built from
// AndroidX and Material Components (Material 3, views); the AAR and the binding stay free of dependencies.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ggtts.demo"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "ggtts.demo"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    // kotlinx-coroutines (which AndroidX brings) ships DebugProbesKt.bin for its debug agent; an app never uses it, and
    // the APK check refuses any .bin as a possible model file.
    packaging { resources { excludes += "DebugProbesKt.bin" } }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AboutTest reads the manifest to hold the screen's "no internet permission" sentence to it, and the screen's tests
// read MainActivity.kt's source. Declared as inputs so a change to either re-runs the tests instead of reusing an
// up-to-date result.
tasks.withType<Test>().configureEach {
    inputs.file("src/main/AndroidManifest.xml").withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file("src/main/kotlin/ggtts/demo/MainActivity.kt").withPathSensitivity(PathSensitivity.RELATIVE)
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(project(":ggtts"))
    implementation("androidx.activity:activity:1.9.3")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    testImplementation("junit:junit:4.13.2")
}
