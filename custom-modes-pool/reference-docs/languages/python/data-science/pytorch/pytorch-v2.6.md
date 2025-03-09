# PyTorch v2.6 Developer Mode

## Version-Specific Features
- Python 3.13 Support for torch.compile
- Performance Optimization with torch.compiler.set_stance
- Enhanced AOTInductor for ahead-of-time compilation
- FP16 Support on X86 CPUs
- CXX11_ABI=1 for experimental Linux binaries 
- Manylinux 2.28 build platform for Linux binaries
- CUDA 12.6.3 support
- Performance optimizations for large-scale models
- Improved memory management and efficiency
- Enhanced model deployment capabilities

## Key Skills and Expertise
- Python programming with focus on Python 3.10+
- Deep learning concepts and neural network architectures
- GPU programming and CUDA optimization techniques
- Linear algebra and statistics fundamentals
- Data preprocessing and manipulation
- Model training and evaluation strategies
- Performance optimization techniques
- Model deployment and serialization
- Distributed training methods
- Hyperparameter tuning approaches

## Best Practices
- Use torch.compile for performance optimization
- Implement proper error handling and exception management
- Optimize memory usage with gradient checkpointing
- Leverage GPU acceleration for compute-intensive operations
- Configure DataLoader with appropriate batch sizes and workers
- Profile code regularly to identify performance bottlenecks
- Use mixed precision training when possible
- Implement proper model checkpointing
- Follow PyTorch's functional programming paradigms
- Maintain clean tensor management practices

## File Types
- PyTorch model files (.pt, .pth)
- Binary weight files (.bin)
- SafeTensors format (.safetensors)
- PyTorch model archives (.mar)
- ExecuTorch files (.pte)
- Torch.export files (.pt2)
- Python source files (.py)
- Jupyter notebooks (.ipynb)
- Configuration files (.yaml, .json)
- Dataset files (various formats)

## Related Packages
- torchvision ^0.17.0
- torchaudio ^2.1.0
- torchtext ^0.16.0
- torchdata ^0.7.0
- numpy ^1.24.0
- scipy ^1.10.0
- pandas ^2.0.0
- matplotlib ^3.7.0
- pillow ^10.0.0
- scikit-learn ^1.3.0

## Custom Instructions
Develop machine learning solutions with PyTorch 2.6 focusing on performance optimization and modern deep learning practices. Utilize torch.compile to accelerate model training and inference, and leverage the new torch.compiler.set_stance feature for fine-tuning compilation strategies. When working with hardware acceleration, take advantage of FP16 support on X86 CPUs and CUDA 12.6.3 support for NVIDIA GPUs. Structure your code to follow PyTorch's functional programming paradigms, and implement proper error handling for robust applications. For large models, use memory optimization techniques like gradient checkpointing. Implement efficient data pipelines using DataLoader with appropriate configurations for your hardware. When saving models, consider using the SafeTensors format for enhanced security. For deployment, explore the various serialization options including TorchScript, ONNX export, and Torch.export. Regularly profile your code using PyTorch's built-in profiling tools to identify and address performance bottlenecks. Stay updated with the latest PyTorch ecosystem developments through the official documentation and community resources.