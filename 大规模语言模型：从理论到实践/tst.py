import pandas as pd
l = [1.0,2.0,3.0,4.0,5.0]
def find_max_smaller_index(arr, target):
    left, right = 0,len(arr) - 1
    max_score = arr[right]
    if pd.isna(target):
        return 11
    if max_score < target:
        return len(arr)-1
    result = 11  # 默认设置为-1，表示没有找到小于target的数
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] > target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result


def find_index_with_binary_search(thresholds, value):
    # 确保thresholds列表是排序的（在这个例子中，它已经是排序的）
    left = 0
    right = len(thresholds) - 1

    # 当区间不为空时继续搜索
    while left <= right:
        mid = (left + right) // 2

        # 如果值小于等于当前阈值，则它属于左侧区间（包括mid）
        if value <= thresholds[mid]:
            # 如果mid不是第一个元素，并且值大于前一个阈值，则它属于(thresholds[mid-1], thresholds[mid]]区间
            if mid > 0 and value > thresholds[mid - 1]:
                return mid
                # 否则，它属于[thresholds[0], thresholds[mid]]区间，但我们返回mid-1（除非mid是0）
            return mid - 1 if mid > 0 else 0
            # 如果值大于当前阈值，则在右侧区间继续搜索
        else:
            left = mid + 1

            # 如果value大于所有阈值，则返回最后一个索引（即列表长度）
    return len(thresholds)


def map_value_to_index(thresholds_, value):
    # thresholds 应该是一个列表，包含了各个阈值点
    # 在这个例子中，我们假设 thresholds = [1.0, 2.0, 3.0, 4.0, float('inf')]
    # 使用 float('inf') 作为无穷大，以确保任何大于最后一个阈值的值都会被映射到最后一个索引
    thresholds = thresholds_ + [float('inf')]  # 添加无穷大作为最后一个阈值

    # 使用 enumerate 来同时获取索引和阈值
    for i, threshold in enumerate(thresholds[:-1]):  # 遍历到倒数第二个阈值，因为最后一个阈值是无穷大
        if value <= threshold:
            return i

            # 如果值大于所有阈值，返回最后一个索引（即列表长度减一）
    return len(thresholds) - 2  # 因为我们添加了一个额外的无穷大阈值，所以索引要减2

print(
    map_value_to_index(
        [],
        1)
)