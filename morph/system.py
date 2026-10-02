"""
pairwise-morph-system - MorphSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class MorphSystem:
    def __init__(self):
        self.state = {}

    def set_morph_weight(self, *args, **kwargs):
        """BUG: 设置Morph权重不做范围限制，权重可以是负数或超过1"""
        return None

    def apply_morph(self, *args, **kwargs):
        """BUG: 顶点变形只做位置偏移，不做法线和切线"""
        return None

    def combine_expressions(self, *args, **kwargs):
        """BUG: 表情组合不做冲突检测，笑和哭同时触发"""
        return None

    def sync_morph_animation(self, *args, **kwargs):
        """BUG: Morph动画不做时间同步，和骨骼动画各跑各的"""
        return None

    def load_morph_data(self, *args, **kwargs):
        """BUG: Morph数据不做压缩，一个角色几十MB"""
        return None

