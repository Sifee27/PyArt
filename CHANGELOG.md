# PyArt Changelog

## v2.1.0 - May 2025
- **Fixed**: Removed hardcoded file paths from batch files (thanks to the Hack Club reviewers for catching this!)
- **Added**: Cross-platform shell scripts for Unix/Linux/macOS users
- **Added**: Proper setup scripts with error handling
- **Improved**: README with better installation instructions
- **Fixed**: All indentation errors that were causing crashes
- **Enhanced**: Emoji system now generates emojis programmatically instead of loading PNGs
- **Fixed**: OpenCV dtype issues in neon_glow and watercolor effects

## v2.0.0 - March 2025
- **Major**: Added voice commands (say "next effect" or "take photo")
- **Added**: Face tracking with emoji overlays (surprisingly fun)
- **Added**: 20+ new visual effects including watercolor, neon glow, kaleidoscope
- **Added**: Motion trails and time-lapse recording
- **Added**: Drawing mode with mouse controls
- **Added**: Split-screen before/after comparison
- **Enhanced**: Hand gesture recognition with MediaPipe

## v1.5.0 - February 2025
- **Added**: Video recording functionality
- **Added**: Burst photo mode for action shots
- **Improved**: Performance optimizations for real-time processing
- **Added**: Color themes and effect blending

## v1.0.0 - January 2025
- **Initial**: Basic ASCII art conversion
- **Added**: Hand gesture controls (thumbs up/down, fist)
- **Added**: Live webcam feed processing
- **Added**: Multiple ASCII character sets

## Known Issues
- Voice commands occasionally get confused if you have a strong accent
- Face tracking can get a bit wonky with multiple people
- Some effects are pretty resource-intensive on older machines

## Planned Features
- Web interface (maybe using Flask)
- More emoji options
- Custom effect creation tools
- Better gesture recognition
