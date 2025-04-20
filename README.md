# IsaacLab Custom Direct envs

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-blue)](https://developer.nvidia.com/isaac-sim)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-green)](https://github.com/NVIDIA-Omniverse/IsaacSim-ros_workspaces)

This repository provides examples for adding **custom direct environments** to IsaacLab. It includes scripts and configurations to help you set up these environments effectively. Follow these examples to extend IsaacLab's capabilities for your specific needs.



## Registor envs
Easily add new environments to IsaacLab using **symbolic links**. This method allows access to files or directories from multiple locations without duplication, simplifying environment management.

### How to Use @make_symbolic.sh

The `make_symbolic.sh` script creates symbolic links for all folders in the current directory. You can use this script to add new environments to the `source/isaaclab_tasks/isaaclab_tasks/direct/` directory in IsaacLab.

#### Usage:

1. Open a terminal and navigate to the directory containing the `make_symbolic.sh` script.
2. Run the script with the following command:
   ```bash
   ./make_symbolic.sh /path/to/isaaclab
   ```
   Here, `/path/to/isaaclab` is the path to the root directory of IsaacLab. If you omit this argument, the current directory will be used as the default path.

3. The script will create symbolic links for all folders in the current directory and add them to the `source/isaaclab_tasks/isaaclab_tasks/direct/` directory.

By following this process, you can easily add new environments to IsaacLab and manage them more efficiently.

## Example command

Currently, we have only created the go1 environment, and an example command for it is provided below.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Unitree-Go1-Direct-v0 --video --video_interval 500 --video_length 250 --num_envs 4096
```