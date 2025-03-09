# .NET MAUI v9.0 Developer Mode

## Version-Specific Features
- **Cross-Platform UI Framework** - Single codebase targeting iOS, Android, macOS, and Windows
- **HybridWebView Control** - New control for integrating HTML/JS/CSS content with C# code communication bridge
- **TitleBar Control** - Customizable app top bar with control over appearance, buttons, and color schemes
- **Enhanced Performance** - Reduced memory usage and faster rendering with expanded test coverage
- **Blazor Desktop Integration** - Improved integration of Blazor components in MAUI applications with BlazorWebView
- **New Soft Keyboard Input Types** - Support for Password, Date, and Time input types on Editor and Entry controls
- **Asset Packs for Android** - Support for placing assets in separate packages up to 2GB in size
- **Improved Text Formatting** - Added 'Justify' option to TextAlignment enumeration for enhanced text layout
- **Enhanced WebView** - Added ProcessTerminated event for handling unexpected web process terminations
- **Upgraded Minimum OS Requirements** - Updated to iOS/Mac Catalyst 15.0 for better performance and feature support

## Key Skills and Expertise
- **C# Programming** with modern language features
- **XAML Design** for declarative UI definition
- **MVVM Architecture** patterns and implementation
- **Cross-Platform Development** techniques for multiple operating systems
- **Platform-Specific Integration** for native functionality
- **Dependency Injection** for service management
- **Responsive UI Design** for different screen sizes
- **Web Technologies** (HTML, CSS, JavaScript) for HybridWebView usage
- **Application Lifecycle Management** across platforms
- **Asynchronous Programming** patterns and practices

## Best Practices
- Use HybridWebView for integrating web content efficiently with native functionality
- Leverage the new TitleBar control for better UI customization and branding
- Utilize the new soft keyboard input types for improved user experience
- Implement text justification where appropriate using the new alignment option
- Handle WebView process terminations using the new event for better error recovery
- For Android apps with large assets, consider using Asset Packs to optimize distribution
- Use MVVM pattern with data binding for clean architecture
- Implement platform-specific services with dependency injection
- Prefer Shell apps for better startup performance and navigation consistency
- Optimize ListView performance by caching data and using efficient cell templates

## File Types
- C# source files (.cs)
- XAML files (.xaml)
- Project files (.csproj)
- Solution files (.sln)
- Resource files (.resx)
- Image files (.png, .jpg, .svg)
- Font files (.ttf)
- HTML/CSS/JS files (for HybridWebView)
- Configuration files (appsettings.json)
- Asset catalogs (for iOS)
- Android resource files (for Android)

## Related Packages
- Microsoft.Maui.Controls ^9.0.0
- Microsoft.Maui.Controls.Compatibility ^9.0.0 (if needed for legacy support)
- Microsoft.Maui.Essentials ^9.0.0
- Microsoft.Extensions.DependencyInjection ^9.0.0
- Microsoft.Extensions.Logging ^9.0.0
- CommunityToolkit.Maui ^9.0.0
- CommunityToolkit.Mvvm ^9.0.0
- Microsoft.Maui.Graphics ^9.0.0
- SkiaSharp.Views.Maui ^6.0.0
- Microsoft.Extensions.Logging.Debug ^9.0.0

## Differences From Previous Version
- **New APIs**:
  - HybridWebView control for hosting web content with native code communication
  - TitleBar control for app title bar customization
  - New soft keyboard input types for Editor and Entry controls
  - TimePicker enhancements with new TimeSelected event
  - WebView improvements with ProcessTerminated event
  
- **Enhanced Features**:
  - Changes in focus behavior on Windows to align with other platforms
  - Addition of 'Justify' text alignment option
  - Introduction of Asset Packs for Android
  - Updated minimum OS version requirements (iOS/Mac Catalyst 15.0)
  - Default to unpackaged app deployment on Windows
  - Performance improvements and expanded testing
  - Improved Visual Studio integration with better Hot Reload

## Custom Instructions
When working with .NET MAUI 9.0, focus on leveraging its cross-platform capabilities and new features to create efficient, responsive applications. Take advantage of the new HybridWebView control to integrate web content seamlessly with your native application, enabling communication between JavaScript in the web view and your C# code. This is particularly valuable when incorporating existing web applications or web-based visualizations. Utilize the new TitleBar control to create a consistent, branded experience across platforms while maintaining native look and feel. For complex data presentation, optimize your ListView performance using the recommended caching techniques and efficient cell templates. Structure your application following the MVVM pattern, continuing to use the CommunityToolkit.Mvvm package for implementation of commands, observable properties, and messaging. With the updated platform minimum requirements (iOS/Mac Catalyst 15.0), ensure you're leveraging the latest platform capabilities while maintaining backward compatibility where required. When targeting Android with large assets, implement the new Asset Packs feature to manage your application size efficiently. For responsive layouts, continue using Grid, FlexLayout, and relative sizing to adapt to different screen sizes and orientations. Take advantage of the enhanced soft keyboard input support for specialized entry fields, and implement text justification where it improves readability. When working with WebView components, always handle the new ProcessTerminated event to provide a graceful recovery path for users in case of unexpected terminations.