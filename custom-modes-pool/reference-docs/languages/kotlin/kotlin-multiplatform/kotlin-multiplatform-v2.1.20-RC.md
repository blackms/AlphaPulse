# Kotlin Multiplatform v2.1.20-RC Developer Mode

## Version-Specific Features
- **Common Atomic Types** - New atomic types in kotlin.concurrent.atomics package for thread-safe operations
- **Improved UUID Handling** - Enhanced parsing and formatting functions with support for multiple formats
- **New Gradle Application DSL** - Replacement for Gradle's Application plugin with Kotlin-specific features
- **K2 Compiler kapt Plugin** - Enabled by default for all projects with improved performance
- **Cross-Platform Code Sharing** - Write once, deploy everywhere for Android, iOS, Web, and desktop
- **Native Interoperability** - Seamless integration with platform-specific APIs and libraries
- **Gradle 8.11 Support** - Compatibility with the latest stable Gradle version
- **Enhanced IDE Integration** - Improved development experience in IntelliJ IDEA and Android Studio
- **Android Gradle Plugin Compatibility** - Support for versions 7.4.2–8.7.2
- **Xcode 16.0 Support** - Seamless integration with the latest Apple development tools

## Key Skills and Expertise
- **Kotlin Programming** with modern language features
- **Gradle Build System** configuration and customization
- **Multiplatform Architecture Design** for code sharing
- **Mobile Development** for Android and iOS
- **Concurrent Programming** with atomic operations and thread safety
- **Native Platform Integration** for platform-specific functionality
- **Dependency Management** across multiple platforms
- **Testing Strategies** for cross-platform code
- **CI/CD Configuration** for multiplatform projects
- **Interoperability** with Java, Swift, JavaScript, and C/C++

## Best Practices
- Use common atomic types for thread-safe operations across platforms
- Leverage the improved UUID parsing and formatting functions for consistent handling
- Migrate to the new DSL for Gradle's Application plugin replacement
- Implement proper dependency injection for platform-specific code
- Structure projects with clear separation between shared and platform-specific code
- Use expect/actual pattern for platform-specific implementations
- Leverage coroutines for asynchronous code that works across platforms
- Test code on all target platforms regularly during development
- Keep platform-specific code to a minimum
- Utilize kotlinx libraries for cross-platform functionality

## File Types
- Kotlin source files (.kt)
- Kotlin script files (.kts, especially build.gradle.kts)
- Gradle properties files (.properties)
- Gradle build files (.gradle)
- Kotlin Multiplatform module configuration files
- Swift source files (.swift) for iOS interoperability
- Java source files (.java) for JVM interoperability
- JavaScript/TypeScript files (.js, .ts) for web interoperability
- C/C++ header files (.h, .hpp) for native interoperability
- Resource files for various platforms

## Related Packages
- Kotlin Standard Library ^2.1.20-RC
- Kotlin Gradle Plugin ^2.1.20-RC
- kotlinx.coroutines (compatible version)
- kotlinx.serialization (compatible version)
- kotlinx.atomicfu (for atomic operations)
- Kotlin Multiplatform Mobile Plugin
- Jetpack Compose Multiplatform
- SQLDelight for cross-platform database access
- Ktor for cross-platform networking
- Koin or Kodein for dependency injection

## Differences From Previous Version
- **New APIs**:
  - Common atomic types in kotlin.concurrent.atomics package
  - Improved UUID parsing and formatting functions
  - New DSL for replacing Gradle's Application plugin
  
- **Compiler Enhancements**:
  - K2 implementation of kapt enabled by default
  - Performance improvements in multiplatform compilation
  - Better error messages and diagnostics
  
- **Platform Support**:
  - Expanded compatibility with latest Android, iOS, and web platforms
  - Improved interoperability with platform-specific libraries
  - Better tooling integration with IDEs and build systems

## Custom Instructions
When working with Kotlin Multiplatform 2.1.20-RC, focus on leveraging its key strengths: code sharing between platforms while maintaining access to platform-specific APIs. This version introduces important improvements for concurrent programming with the new common atomic types in the kotlin.concurrent.atomics package, allowing you to write thread-safe code once and deploy it across platforms. Take advantage of the improved UUID parsing and formatting functions which now support both hex-and-dash and plain hexadecimal formats through the new parseHexDash() and toHexDashString() methods. When setting up your build system, explore the new DSL that replaces Gradle's Application plugin, providing a more Kotlin-centric approach to configuring your applications. Structure your multiplatform projects with a clear separation between common code and platform-specific implementations using the expect/actual pattern. For asynchronous programming, utilize kotlinx.coroutines which works seamlessly across platforms. When implementing platform-specific features, keep the interface in the common code and only implement the necessary platform details in the actual implementations. For persistence, consider cross-platform solutions like SQLDelight, and for networking, use Ktor which provides a consistent API across platforms. Remember that since this is a release candidate (RC), you should be prepared for potential changes before the final release and report any issues you encounter to help improve the final version. Also note that the K2 implementation of the kapt compiler plugin is now enabled by default, which should provide performance improvements but might require some adjustments if you were previously using the older implementation.