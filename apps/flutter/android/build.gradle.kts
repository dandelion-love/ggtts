allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// mp_audio_stream 0.3.0 pins an obsolete SDK and NDK. Match the app so its AndroidX
// dependencies compile and its native library uses the same 16 KB-capable toolchain.
subprojects {
    if (name == "mp_audio_stream") {
        afterEvaluate {
            val appAndroid = project(":app").extensions.getByType<com.android.build.api.dsl.ApplicationExtension>()
            extensions.getByType<com.android.build.api.dsl.LibraryExtension>().apply {
                compileSdk = appAndroid.compileSdk
                ndkVersion = appAndroid.ndkVersion
                defaultConfig.minSdk = appAndroid.defaultConfig.minSdk
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
