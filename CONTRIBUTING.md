# Contributing to PyArt

Thank you for your interest in contributing to PyArt! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test your changes
5. Commit your changes: `git commit -m 'Add some feature'`
6. Push to the branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## Development Setup

1. Clone your forked repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python test_installation.py`

## Adding New Effects

To add a new effect:

1. Open `src/effects.py`
2. Add a new method to the `EffectProcessor` class
3. Register your effect in the `__init__` method
4. Test your effect using the demo mode: `python demo.py`

## Pull Request Process

1. Update the README.md with details of changes if appropriate
2. Update the demo.py file if you've added new functionality
3. The PR will be merged once reviewed and approved

## Style Guidelines

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Include docstrings for all functions, classes, and methods
- Keep code modular and maintainable

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
