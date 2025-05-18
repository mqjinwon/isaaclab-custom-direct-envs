# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# =============================== #
# RSL-RL Flat
# =============================== #

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-Direct-v0",
    entry_point=f"{__name__}.unitree_go1_env:UnitreeGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go1_cfg:UnitreeGo1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-Direct-Play-v0",
    entry_point=f"{__name__}.unitree_go1_env:UnitreeGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go1_cfg:UnitreeGo1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)


# =============================== #
# RSL-RL Rough
# =============================== #

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Direct-v0",
    entry_point=f"{__name__}.unitree_go1_env:UnitreeGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go1_cfg:UnitreeGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Direct-Play-v0",
    entry_point=f"{__name__}.unitree_go1_env:UnitreeGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go1_cfg:UnitreeGo1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# =============================== #
# Dreamwaq
# =============================== #

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Dreamwaq-Direct-v0",
    entry_point=f"{__name__}.unitree_go1_env:UnitreeGo1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_go1_cfg:UnitreeGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1DreamwaqPPORunnerCfg",
    },
)