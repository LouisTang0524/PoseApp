"""
Repetition counter
"""

class RepetitionCounter(object):
    # 计算给定目标姿势类的重复次数

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name = class_name

        # 如果姿势通过了给定的阈值，就进入该动作的计数
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # 是否处于给定的姿势
        self._pose_entered = False

        # 退出姿势的次数
        self._n_repeats = 0
    
    @property
    def n_repeats(self):
        return self._n_repeats
    
    def __call__(self, pose_classification):
        # 计算给定帧之间发生的重复次数
        # 给定两个阈值，从较高的位置上方进入，从较低的位置下方退出
        # 阈值之间的差异使其对预测的抖动稳定

        # Args:
        #     pose_classification: 当前帧上的姿势分类的字典
        #         {
        #             'squat_down': 8.3, 置信度
        #             'squat_up': 1.7,
        #         }

        # 获取姿势的置信度
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]
        
        # 在第一帧或如果不处于姿势中，只需检查我们是否在这一帧上进入该姿势并更新状态
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats
        
        # 如果处于姿势中并且正在推出它，则增加计数器并更新状态
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False
        
        return self._n_repeats
