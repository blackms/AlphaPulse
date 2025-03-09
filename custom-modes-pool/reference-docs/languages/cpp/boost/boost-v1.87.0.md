# Boost v1.87.0 Developer Mode

## Version-Specific Features
- **Comprehensive C++ Library Collection** - 165+ individual libraries covering virtually every aspect of C++ development
- **C++ Standards Compliance** - Fully compatible with modern C++11/14/17/20 standards and newer compiler versions
- **Generic Dispositions Support** - New types for testing whether asynchronous operations completed without errors
- **Metaprogramming Utilities** - Advanced template metaprogramming tools including Boost.MPL and Boost.Hana
- **Smart Pointers** - Memory management utilities like shared_ptr, weak_ptr, and intrusive_ptr
- **Containers and Data Structures** - Specialized containers like Boost.MultiIndex and Boost.Container
- **Asynchronous I/O** - Boost.Asio library for network and low-level I/O programming
- **Multithreading Support** - Concurrency utilities with Boost.Thread and Boost.Fiber
- **File System Operations** - Portable filesystem operations with Boost.Filesystem
- **Parser Library** - New library for parsing tasks added in this version

## Key Skills and Expertise
- **Advanced C++ Programming** with template metaprogramming
- **Modern C++ Idioms** using C++11/14/17/20 features
- **Generic Programming** techniques with templates
- **Memory Management** patterns and practices
- **Functional Programming** with Boost.Function and Boost.Lambda
- **Concurrent Programming** using Boost.Thread and Boost.Asio
- **Network Programming** with Boost.Asio and Boost.Beast
- **Error Handling** with Boost.Exception and error codes
- **Serialization** using Boost.Serialization
- **Testing** with Boost.Test framework

## Best Practices
- Use header-only libraries when possible for easier integration
- Leverage Boost.TypeTraits or C++ standard library type traits instead of deprecated features
- Utilize the bcp utility to extract subsets of Boost for distribution with applications
- Use Boost libraries as a foundation for portable, standards-compliant code
- Leverage Boost.SmartPtr for safer resource management
- Implement asynchronous operations with Boost.Asio for better scalability
- Utilize Boost.Container for specialized data structures not available in the standard library
- Implement proper error handling with Boost.Exception
- Use Boost.Bimap and Boost.MultiIndex for complex data relationships
- Implement comprehensive testing with Boost.Test

## File Types
- C++ header files (.h, .hpp)
- C++ source files (.cpp, .cc)
- Boost library headers (.hpp)
- CMake configuration files (CMakeLists.txt)
- Boost Jam files (Jamfile, Jamroot)
- Build configuration files (build.jam)
- Documentation files (.md, .html)
- Test files (_test.cpp)
- Example files (_example.cpp)
- Quickbook and Doxygen documentation formats

## Related Packages
- Boost.Asio ^1.87.0
- Boost.Beast ^1.87.0
- Boost.Filesystem ^1.87.0
- Boost.Thread ^1.87.0
- Boost.Container ^1.87.0
- Boost.Graph ^1.87.0
- Boost.Test ^1.87.0
- Boost.MultiIndex ^1.87.0
- Boost.Serialization ^1.87.0
- Boost.JSON ^1.87.0
- Python (optional, for Python bindings)
- ICU library (for build)
- OpenMPI (for build)
- zlib (for build)
- zstd (for build)

## Differences From Previous Version
- **New APIs**:
  - Generic dispositions for asynchronous operation testing
  - New Parser library added to the collection
  - Updates to 30+ existing libraries including Asio, Beast, Container, and JSON
  
- **Enhanced Features**:
  - Improved support for recent compiler versions (Clang 15.0.0, GCC 12)
  - Better compatibility with C++20 features
  - Enhanced performance in core libraries
  - Improved documentation and examples
  - More comprehensive testing utilities

## Custom Instructions
When working with Boost 1.87.0, focus on leveraging its comprehensive suite of C++ libraries to solve complex programming challenges with well-tested, portable code. This version supports the latest compilers and C++ standards while maintaining backward compatibility. Consider using header-only libraries when possible to simplify integration, and utilize the bcp utility when you need to distribute Boost components with your application. For memory management, prefer Boost's smart pointer implementations for legacy code or when you need specialized behavior like intrusive_ptr. For networking and asynchronous I/O, take advantage of the new generic dispositions support in Boost.Asio for more robust error handling in event-driven programming. When dealing with complex data structures, consider Boost.MultiIndex for maintaining multiple indices over the same dataset, and Boost.Bimap for bidirectional maps. For metaprogramming tasks, leverage Boost.MPL, Boost.Fusion, and Boost.Hana to write highly generic code with strong type safety. Use Boost.Filesystem for portable path manipulation and file operations in codebases that haven't yet migrated to std::filesystem. When implementing concurrent code, use appropriate Boost threading abstractions and synchronization primitives. For error handling, leverage Boost.Exception for richer exception handling with stacktraces and additional context information. Finally, implement comprehensive testing with Boost.Test, taking advantage of its test fixtures, parameterized testing, and detailed reporting capabilities.