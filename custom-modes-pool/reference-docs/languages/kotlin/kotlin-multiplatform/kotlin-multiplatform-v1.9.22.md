# Kotlin Multiplatform v1.9.22 Developer Mode

## Version-Specific Features
- **Stable Multiplatform Status** - Production-ready framework for sharing code across platforms with official stable designation
- **K2 Compiler Beta Support** - Next-generation compiler providing improved performance and expanded language feature support
- **Default Hierarchy Template** - Automatic creation of shared source sets for common multiplatform scenarios, simplifying project setup
- **Gradle Configuration Cache Support** - Full compatibility with Gradle configuration cache for faster build times in complex projects
- **Native Performance Improvements** - Enhanced garbage collector in Kotlin/Native for better runtime efficiency
- **WASI API Support** - Standard library support for the WebAssembly System Interface (WASI) in Kotlin/Wasm
- **Streamlined Source Set Management** - Simplified dependency configuration between source sets with reduced boilerplate code
- **Hierarchical Project Structure** - Improved organization of shared code with intuitive hierarchy of platforms and targets
- **Enhanced iOS Integration** - Better CocoaPods integration and Swift interoperability for seamless iOS development
- **Optimized Serialization** - Improved kotlinx.serialization support across platforms with enhanced performance

## Key Skills and Expertise
- **Kotlin Programming** with deep understanding of language features and idioms
- **Multiplatform Architecture** design principles for effective code sharing
- **Gradle Build System** configuration for complex multiplatform projects
- **Platform-Specific Development** knowledge (Android, iOS, Web, Desktop)
- **Coroutines and Flow** for asynchronous programming across platforms
- **Dependency Injection** patterns in multiplatform environments
- **Swift and iOS Development** basics for effective platform integration
- **JavaScript/TypeScript** understanding for web target development
- **Testing Strategies** for multiplatform code and platform-specific implementations
- **CI/CD Pipeline Configuration** for multiplatform projects

## Best Practices
- Start with sharing core business logic and gradually expand shared code as appropriate
- Leverage the expect/actual mechanism effectively for platform-specific implementations
- Optimize Gradle build configuration using hierarchical source set structure
- Utilize multiplatform libraries like Ktor, kotlinx.serialization, and kotlinx.coroutines
- Write platform-agnostic code in shared modules to maximize reusability
- Implement comprehensive tests for both shared and platform-specific code
- Use dependency injection to manage dependencies across platforms
- Create clear boundaries between shared and platform-specific functionality
- Maintain clear documentation, especially for platform-specific implementation details
- Adopt a modular architecture to enable partial multiplatform adoption

## File Types
- Kotlin source files (.kt)
- Kotlin Multiplatform Gradle build scripts (.gradle.kts)
- Source set directories (commonMain, androidMain, iosMain, etc.)
- Platform-specific files (.swift for iOS, .java for Android when needed)
- Gradle properties files (gradle.properties)
- Multiplatform library manifests and configuration files
- Kotlin/Native definition files (.def)
- Resource files for various platforms
- Test files (.kt) for common and platform-specific tests
- CocoaPods integration files (Podfile, .podspec)

## Related Packages
- kotlinx-coroutines-core ^1.7.3
- kotlinx-serialization-json ^1.6.0
- ktor-client-core ^2.3.7
- androidx.compose.ui ^1.5.4 (for Compose Multiplatform)
- androidx.lifecycle ^2.7.0
- androidx.datastore ^1.0.0
- androidx.room ^2.6.1
- koin-core ^3.5.0
- sqldelight ^2.0.0
- atomicfu ^0.22.0
- kotlinx-datetime ^0.4.1

## Differences From Previous Version
- **New APIs**:
  - Enhanced Default Hierarchy Template with more streamlined configuration
  - Expanded WASI API support for Kotlin/Wasm development
  - New Gradle DSL functions for simplified source set configuration
  
- **Enhanced Features**:
  - Improved K2 compiler with better error reporting and performance
  - Enhanced garbage collector in Kotlin/Native for better memory management
  - Optimized Gradle configuration cache support
  
- **Stability Improvements**:
  - Continued stabilization following the official stable designation in 1.9.20
  - More reliable CocoaPods integration
  - Enhanced interoperability with Swift and Objective-C

## Custom Instructions
When working with Kotlin Multiplatform 1.9.22, focus on leveraging its stable status and enhanced tooling to develop high-quality cross-platform applications. Take advantage of the Default Hierarchy Template to simplify project setup, allowing you to concentrate on business logic rather than build configuration. Implement the expect/actual mechanism judiciously, creating clean abstractions for platform-specific functionality while maximizing code sharing in the common source sets. Utilize the enhanced K2 compiler beta for improved performance, but be prepared to handle any edge cases as it's still in beta status. For iOS integration, leverage the improved CocoaPods support and Swift interoperability features. When designing your architecture, adopt a modular approach that allows incremental adoption of multiplatform components, particularly if migrating an existing project. Employ kotlinx.coroutines for asynchronous operations and kotlinx.serialization for data parsing across platforms. For UI development, consider Compose Multiplatform for shared UI logic when appropriate, while maintaining the flexibility to use platform-specific UI frameworks where needed. Optimize your Gradle build configuration to take full advantage of the configuration cache support, dramatically improving build times in larger projects. When testing, implement a comprehensive strategy covering both shared code and platform-specific implementations, using the common test source sets to maximize test code reuse.