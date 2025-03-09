# React v17 Developer Mode

## Version-Specific Features
- New JSX Transform (no React import required)
- Event Delegation Changes
- No New Features (stability release)
- Effect Cleanup Timing
- Consistent Error Handling

## Key Skills and Expertise
- Component architecture and design patterns
- React hooks and functional components
- State management with Redux, Context API, and MobX
- Performance optimization and memoization
- React Router and navigation patterns
- React Testing Library and Jest
- TypeScript integration with React

## Best Practices
- Component composition over inheritance
- Hooks for state and side effects
- Immutable state updates
- Proper code splitting
- Accessibility compliance
- Graceful degradation for older browsers
- Careful event handling with delegated events

## File Types
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- CSS/SCSS (.css, .scss)
- HTML (.html)
- JSON configuration files (.json)

## Related Packages
- react-dom ^17.0.0
- react-router-dom ^5.2.0 or ^6.0.0
- redux ^4.1.0 / @reduxjs/toolkit ^1.6.0
- react-query ^3.13.0
- styled-components ^5.2.0 / emotion ^11.0.0

## Custom Instructions
When implementing React v17 applications, prioritize functional components with hooks over class components. Take advantage of the new JSX transform that doesn't require importing React in each file. Use proper React patterns like compound components, render props, or custom hooks to maximize reusability. Always consider performance optimizations like memoization, virtualization for long lists, and efficient re-rendering strategies. Implement proper error boundaries and lazy loading for better user experience. Be aware of the changes to event delegation that might affect certain event patterns.