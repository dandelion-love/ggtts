// The demo app: one screen that loads the model files the user picks and speaks one line with either engine, through
// the binding in :ggtts (built as an AAR). No model ships with it; see android/README.md.
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

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AboutTest reads the manifest to hold the screen's "no internet permission" sentence to it. Declared as an input so
// a manifest change re-runs the tests instead of reusing an up-to-date result.
tasks.withType<Test>().configureEach {
    inputs.file("src/main/AndroidManifest.xml").withPathSensitivity(PathSensitivity.RELATIVE)
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(project(":ggtts"))
    testImplementation("junit:junit:4.13.2")
}
