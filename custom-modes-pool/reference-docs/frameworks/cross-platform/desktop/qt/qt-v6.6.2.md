# Qt v6.6.2 Developer Mode

## Version-Specific Features
- **Cross-Platform Framework** - Comprehensive C++ framework for developing applications across desktop, mobile, embedded, and IoT platforms
- **QML and Qt Quick** - Declarative UI system with JavaScript integration for rapid interface development
- **Qt Widgets** - Traditional widget-based UI components for desktop applications
- **Qt WebEngine** - Chromium-based web browser engine for embedding web content
- **Qt 3D** - Framework for 3D visualization and simulations
- **Qt Multimedia** - Audio and video playback and capture functionality
- **CMake Build System** - Modern CMake-based build system replacing QMake
- **Property Bindings** - New property binding system for more reactive UI development
- **State Machine Framework** - Tools for implementing event-driven state machines
- **Qt Network** - Comprehensive networking capabilities including HTTP, WebSockets, and SSL

## Key Skills and Expertise
- **C++ Programming** with modern C++17/20 features
- **Object-Oriented Design** using Qt's metaobject system
- **QML and JavaScript** for declarative UI development
- **Signal-Slot Mechanism** for object communication
- **Event Handling** using Qt's event system
- **Model-View-Controller Pattern** implementation
- **Multithreading** with Qt Concurrent and QThread
- **Cross-Platform Development** techniques
- **GPU Programming** with Qt's OpenGL and Vulkan integrations
- **CMake Build System** configuration and customization

## Best Practices
- Leverage Qt's ownership system with parent-child relationships
- Use Qt's signal-slot mechanism for loose coupling between components
- Implement proper resource management with RAII principles
- Utilize QML for modern, fluid interfaces and Qt Widgets for complex desktop applications
- Take advantage of the property binding system for reactive UIs
- Implement threading carefully, using appropriate synchronization mechanisms
- Follow Qt's naming conventions for better code readability
- Use Qt's model/view architecture for data presentation
- Leverage Qt's internationalization system for localization
- Implement proper error handling with appropriate propagation

## File Types
- C++ header files (.h, .hpp)
- C++ source files (.cpp, .cc)
- QML files (.qml)
- Qt resource files (.qrc)
- Qt user interface files (.ui)
- CMake configuration files (CMakeLists.txt)
- Qt project files (.pro) (legacy)
- Qt Linguist translation files (.ts, .qm)
- JSON configuration files (.json)
- QSS style sheets (.qss)

## Related Packages
- Qt Base ^6.6.2
- Qt Quick ^6.6.2
- Qt 3D ^6.6.2
- Qt WebEngine ^6.6.2
- Qt Multimedia ^6.6.2
- Qt Network ^6.6.2
- Qt Charts ^6.6.2
- Qt DataVisualization ^6.6.2
- Qt Concurrent ^6.6.2
- Qt Positioning ^6.6.2

## Differences From Previous Version
- **New APIs**:
  - Enhanced property binding system with more reactive capabilities
  - Improved QML integration with C++ types
  - New network features for modern protocols
  
- **Enhanced Features**:
  - Better CMake integration and build system support
  - Improved performance in Qt Quick rendering
  - Enhanced threading and concurrency utilities
  - More comprehensive WebEngine capabilities
  - Better support for modern C++ features

## Custom Instructions
When working with Qt 6.6.2, focus on leveraging its comprehensive cross-platform capabilities to build modern, responsive applications. This version continues Qt's evolution with improvements to the property binding system, CMake integration, and QML engine. Structure your applications following Qt's object ownership model, where parent objects take ownership of their children, ensuring proper resource management. For UI development, choose between QML (for modern, touch-friendly interfaces with fluid animations) and Qt Widgets (for traditional desktop applications with complex controls). Take advantage of Qt's signal-slot mechanism for communication between objects, enabling loose coupling and maintainable code. For multithreading, prefer higher-level abstractions like Qt Concurrent when possible, falling back to QThread for more complex scenarios with appropriate synchronization. Utilize Qt's model/view architecture for efficiently displaying and manipulating data. When developing cross-platform applications, abstract platform-specific code behind interfaces and leverage Qt's platform abstraction layers. For building, embrace the CMake-based build system which offers better integration with modern C++ development workflows and third-party libraries. Use Qt's resource system (QRC files) to embed assets directly into your application binary for easier deployment. For networking, take advantage of Qt's comprehensive networking classes which provide high-level abstractions for various protocols while handling platform differences transparently.