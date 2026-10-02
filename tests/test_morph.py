import pytest
from morph.system import MorphSystem


class TestMorphSystem:

    def test_set_morph_weight(self):
        """设置Morph权重不做范围限制，权重可以是负数或超过1"""
        obj = MorphSystem()
        result = obj.set_morph_weight()
        self.assertIsNotNone(result)

    def test_apply_morph(self):
        """顶点变形只做位置偏移，不做法线和切线"""
        obj = MorphSystem()
        result = obj.apply_morph()
        self.assertIsNotNone(result)

    def test_combine_expressions(self):
        """表情组合不做冲突检测，笑和哭同时触发"""
        obj = MorphSystem()
        result = obj.combine_expressions()
        self.assertIsNotNone(result)

    def test_sync_morph_animation(self):
        """Morph动画不做时间同步，和骨骼动画各跑各的"""
        obj = MorphSystem()
        result = obj.sync_morph_animation()
        self.assertIsNotNone(result)

    def test_load_morph_data(self):
        """Morph数据不做压缩，一个角色几十MB"""
        obj = MorphSystem()
        result = obj.load_morph_data()
        self.assertIsNotNone(result)

