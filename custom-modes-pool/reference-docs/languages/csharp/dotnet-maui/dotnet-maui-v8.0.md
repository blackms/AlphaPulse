# .NET MAUI v8.0 Developer Mode

## Version-Specific Features
- **Cross-Platform UI Framework** - Single codebase targeting iOS, Android, macOS, and Windows
- **Blazor Desktop Integration** - Use Blazor components in MAUI applications with BlazorWebView
- **Graphics and Animation** - Hardware-accelerated graphics with GPU-powered rendering 
- **UI Controls Library** - Comprehensive set of native UI controls with unified API
- **Unified Resource System** - Single project for images, fonts, and other resources across platforms
- **Hot Reload** - Real-time UI updates during development without app restarts
- **Windows App SDK Integration** - Direct access to Windows-specific APIs for advanced functionality
- **Adaptive Layouts** - Responsive UI that adapts to different screen sizes and orientations
- **Multi-Window Support** - Create and manage multiple windows on desktop platforms
- **.NET 8 Foundation** - Built on the performance and features of .NET 8 runtime

## Key Skills and Expertise
- **C# Programming** with modern language features
- **XAML Design** for declarative UI definition
- **MVVM Architecture** patterns and implementation
- **Cross-Platform Development** techniques for multiple operating systems
- **Platform-Specific Integration** for native functionality
- **Dependency Injection** for service management
- **Responsive UI Design** for different screen sizes
- **Application Lifecycle Management** across platforms
- **Animation and Graphics** programming
- **Deployment and Distribution** for multiple app stores

## Best Practices
- Use MVVM pattern with data binding for clean architecture
- Implement platform-specific services with dependency injection
- Leverage Shell navigation for consistent navigation experience
- Utilize resource dictionaries for consistent theming
- Implement responsive layouts with FlexLayout and Grid
- Use Collection View for efficient data presentation
- Leverage semantic properties for accessibility
- Implement proper lifecycle management with handlers
- Optimize image resources for each target platform
- Use conditional compilation for platform-specific code when necessary

## File Types
- C# source files (.cs)
- XAML files (.xaml)
- Project files (.csproj)
- Solution files (.sln)
- Resource files (.resx)
- Image files (.png, .jpg, .svg)
- Font files (.ttf)
- Configuration files (appsettings.json)
- Asset catalogs (for iOS)
- Android resource files (for Android)

## Related Packages
- Microsoft.Maui.Controls ^8.0.0
- Microsoft.Maui.Controls.Compatibility ^8.0.0
- Microsoft.Maui.Essentials ^8.0.0
- Microsoft.Extensions.DependencyInjection ^8.0.0
- Microsoft.Extensions.Logging ^8.0.0
- CommunityToolkit.Maui ^6.0.0
- CommunityToolkit.Mvvm ^8.2.0
- Microsoft.Maui.Graphics ^8.0.0
- SkiaSharp.Views.Maui ^5.0.0
- Sharpnado.Tabs.Maui ^3.0.0

## Differences From Previous Version
- **New APIs**:
  - Enhanced BlazorWebView integration
  - Improved multi-window support
  - New controls for desktop scenarios
  
- **Enhanced Features**:
  - Better performance with .NET 8 runtime
  - Improved startup time across all platforms
  - Enhanced developer tooling and debugging
  - Better Windows App SDK integration
  - More consistent behavior across platforms

## Custom Instructions
When working with .NET MAUI 8.0, focus on leveraging its cross-platform capabilities to create native applications with a single codebase. Structure your application following the MVVM pattern, using the CommunityToolkit.Mvvm package for robust implementation of commands, observable properties, and messaging. Take advantage of MAUI's Shell navigation system for a consistent navigation experience across platforms, defining your application's navigational hierarchy declaratively in XAML. For UI design, prefer using modern MAUI controls like CollectionView and CarouselView over legacy Xamarin.Forms controls for better performance and flexibility. Utilize the Handler architecture for platform-specific customizations, allowing you to modify control appearance and behavior on specific platforms without breaking the shared abstraction. Implement proper dependency injection using the built-in Microsoft.Extensions.DependencyInjection container, registering platform-specific implementations of interfaces for services that need to access native functionality. For responsive layouts, combine Grid, FlexLayout, and relative sizing to ensure your UI adapts well to different screen sizes and orientations. Take advantage of the unified resource system to manage images, fonts, and other assets efficiently across platforms. When performance is critical, consider using .NET MAUI Graphics for custom drawing and animations that leverage hardware acceleration. For advanced scenarios on Windows, utilize Windows App SDK integration to access platform-specific capabilities while maintaining cross-platform compatibility for your core functionality. Always test your application thoroughly on all target platforms, as subtle platform differences can impact user experience despite the unified API.