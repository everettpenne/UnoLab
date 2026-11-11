plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing: the Uno Lab key, from these four variables (see scripts/release-signed.sh). Without them a release build is signed with the
// debug key, for local use only.
val releaseVariables = listOf("UNOLAB_RELEASE_STORE_FILE", "UNOLAB_RELEASE_STORE_PASSWORD", "UNOLAB_RELEASE_KEY_ALIAS", "UNOLAB_RELEASE_KEY_PASSWORD")
val releaseValues = releaseVariables.associateWith { System.getenv(it)?.takeIf { v -> v.isNotBlank() } }
val releaseSigning = releaseValues.values.all { it != null }

android {
    namespace = "com.everettpenne.unolab"
    compileSdk = 36
    ndkVersion = "26.3.11579264"
    defaultConfig {
        applicationId = "com.everettpenne.unolab"
        minSdk = 31
        targetSdk = 36
        versionCode = 7
        versionName = "0.3.4"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_STL=c++_shared") } }
    }
    signingConfigs {
        if (releaseSigning) create("unolab") {
            storeFile = file(releaseValues.getValue("UNOLAB_RELEASE_STORE_FILE")!!)
            storePassword = releaseValues.getValue("UNOLAB_RELEASE_STORE_PASSWORD")
            keyAlias = releaseValues.getValue("UNOLAB_RELEASE_KEY_ALIAS")
            keyPassword = releaseValues.getValue("UNOLAB_RELEASE_KEY_PASSWORD")
        }
    }
    buildTypes {
        // Inference is unusably slow in a debuggable build, so the debug build is not debuggable and its native code is optimised.
        getByName("debug") { isDebuggable = false; isJniDebuggable = false; externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } } }
        getByName("release") { isMinifyEnabled = false; signingConfig = signingConfigs.getByName(if (releaseSigning) "unolab" else "debug") }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    buildFeatures { compose = true; aidl = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    packaging { jniLibs { useLegacyPackaging = false } }
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.06.01"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.1")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
