package binarysearch

import "math"

/*
	二分查找：
		二分查找一般由三个主要部分组成：
			1、预处理：
				如果集合未排序，则进行排序。
			2、二分查找：
				使用循环或递归在每次比较后将查找空间划分为两半。
			3、后处理：
				在剩余空间中确定可行的候选者。
		常用的有三个模板：
			1、(left <= right)
				每次循环查找空间只有一个元素（nums[mid]），
				循环结束（left == right + 1）即查找完成，不需要做后续处理。
			2、(left < right)
				每次循环查找空间有两个元素（nums[mid]、nums[mid+1]），
				循环结束时（left == right）会剩余一个元素 num[left]，需要做后续处理。
			3、(left + 1 < right)
				每次循环查找空间有三个元素（nums[mid - 1]、nums[mid]、nums[mid+1]），
				循环结束时（left + 1 == right）会剩余两个元素 nums[left]和nums[right]，需要做后续处理。
	时间复杂度：O(log n)
		二分查找是通过对查找空间中间的值应用一个条件来操作的，并因此将查找空间折半，
		在更糟糕的情况下，我们将不得不进行 O(log n) 次比较，其中 n 是集合中元素的数目。
	空间复杂度：O(1)
		我们只需要3个变量的额外空间。
*/
/*
===================== 1、二分查找 =====================
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4

示例 2:
输入: nums = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不存在 nums 中因此返回 -1

提示：
    你可以假设 nums 中的所有元素是不重复的。
    n 将在 [1, 10000]之间。
    nums 的每个元素都将在 [-9999, 9999]之间。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-search
*/
/*
	二分查找模板一：(left <= right)
		关键属性：
			二分查找的最基础和最基本的形式。
			查找条件可以在不与元素的两侧进行比较的情况下确定（或使用它周围的特定元素）。
			不需要后处理，因为每一步中，你都在检查是否找到了元素。如果到达末尾，则知道未找到该元素。
		区分语法：
			初始条件：left = 0, right = length-1
			终止条件：left > right
			向左查找：right = mid-1
			向右查找：left = mid+1
			后续处理：无
*/
func binarySearch_1(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n-1
	for L <= R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		// 同 L + (R - L) / 2，加括号是因为算数运算符的优先级大于位运算符
		mid := L + ((R - L) >> 1)
		if nums[mid] == target {
			return mid
		}
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid - 1
		}
	}
	// L == R + 1 时循环停止，此时查找完成，无需后续处理
	return -1
}

/*
	二分查找模板二：(left < right)
		一种实现二分查找的高级方法。
		查找条件需要访问元素的 直接右邻居。
		使用元素的右邻居来确定是否满足条件，并决定是向左还是向右。
		保证查找空间在每一步中至少有 2 个元素。
		需要进行后续处理。
		当你剩下 1 个元素时，循环 / 递归结束。 需要评估剩余元素是否符合条件。
	区分语法：
		初始条件：left = 0, right = n
		终止条件：left = right
		向左查找：right = mid
		向右查找：left = mid + 1
		后续处理：
			if left != n && nums[left] == target {
				return L
			}
*/
func binarySearch_2(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n
	for L < R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		// 同 L + (R - L) / 2，加括号是因为算数运算符的优先级大于位运算符
		mid := L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	// L == R 时循环停止，此时会剩余一个元素需要判断
	if L != n && nums[L] == target {
		return L
	}
	return -1
}

/*
	二分查找模板三：(left + 1 < right)
		实现二分查找的另一种方法。
		搜索条件需要访问元素的直接左、右邻居。
		使用元素的邻居来确定它是向右还是向左。
		保证查找空间在每个步骤中至少有 3 个元素。
		需要进行后处理。 当剩下 2 个元素时，循环 / 递归结束。 需要评估其余元素是否符合条件。
*/
func binarySearch_3(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n-1
	for L+1 < R {
		// 此处不能用(L+R)/2，因为加完后有溢出的风险
		mid := L + (R-L)>>1
		if target > nums[mid] {
			L = mid
		} else {
			R = mid
		}
	}
	// L + 1 == R 时循环停止，此时会剩余两个元素需要判断
	if nums[L] == target {
		return L
	}
	if nums[R] == target {
		return R
	}
	return -1
}

/*
===================== 2、x 的平方根 =====================
实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:
输入: 4
输出: 2

示例 2:
输入: 8
输出: 2
说明: 8 的平方根是 2.82842...,
     由于返回类型是整数，小数部分将被舍去。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/sqrtx
*/
/*
	方法一：二分查找【模板一】
	思路：
		由于 x 平方根的整数部分 ans 是满足 k^2 ≤ x 的最大 k 值，因此我们可以对
		k 进行二分查找，从而得到答案。
		二分查找的下界为 0，上界可以粗略地设定为 x。在二分查找的每一步中，我们只
		需要比较中间元素 mid 的平方与 x 的大小关系，并通过比较的结果调整上下界的
		范围。由于我们所有的运算都是整数运算，不会存在误差，因此在得到最终的答案
		ans 后，也就不需要再去尝试 ans+1 了。
	时间复杂度：O(log(x))
		我们需要对 x 进行二分计算。
	空间复杂度：O(1)
*/
func mySqrt(x int) int {
	L, R := 0, x
	ans := 0
	for L <= R {
		mid := L + ((R - L) >> 1)
		if mid*mid <= x {
			ans = mid
			L = mid + 1
		} else {
			R = mid - 1
		}
	}
	return ans
}

/*
	方法二：袖珍计算器算法
	思路：
		「袖珍计算器算法」是一种用指数函数 exp 和对数函数 ln⁡ 代替平方根函数的方
		法。我们通过有限的可以使用的数学函数，得到我们想要计算的结果。
		我们将 √x 写成幂的形式 x^(1/2)，再使用自然对数 e 进行换底，即可得到
			√x = x^(1/2) = (e^(ln⁡x))^(1/2) = e^((1/2)*ln⁡x)
		这样我们就可以得到 √x 的值了。

		注意： 由于计算机无法存储浮点数的精确值（浮点数的存储方法可以参考
		IEEE 754，这里不再赘述），而指数函数和对数函数的参数和返回值均为浮点数，
		因此运算过程中会存在误差。例如当 x=2147395600 时，e^((1/2)*ln⁡x) 的计算
		结果与正确值 463404634046340 相差 10^(−11)，这样在对结果取整数部分时，
		会得到 463394633946339 这个错误的结果。

		因此在得到结果的整数部分 ans 后，我们应当找出 ans 与 ans+1 中哪一个是
		真正的答案。
	时间复杂度：O(1)
		由于内置的 exp 函数与 log 函数一般都很快，我们在这里将其复杂度视为 O(1)。
	空间复杂度：O(1)。
*/
func mySqrt(x int) int {
	if x == 0 {
		return 0
	}
	ans := int(math.Exp(0.5 * math.Log(float64(x))))
	if (ans+1)*(ans+1) <= x {
		return ans + 1
	}
	return ans
}

/*
===================== 3、猜数字大小 =====================
猜数字游戏的规则如下：
    每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
    如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。

你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一
共有 3 种可能的情况（-1，1 或 0）：
    -1：我选出的数字比你猜的数字小 pick < num
    1：我选出的数字比你猜的数字大 pick > num
    0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num

返回我选出的数字。

示例 1：
输入：n = 10, pick = 6
输出：6

示例 2：
输入：n = 1, pick = 1
输出：1

示例 3：
输入：n = 2, pick = 1
输出：1

示例 4：
输入：n = 2, pick = 2
输出：2

提示：
    1 <= n <= 231 - 1
    1 <= pick <= n

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xee4ev/
*/
/*
	方法一：二分查找【模板一】
	思路：
		以 n 作为上限，根据 guess(int num) 的返回结果进行二分查找。
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func guessNumber(n int) int {
	L, R := 0, n
	for L <= R {
		mid := L + ((R - L) >> 1)
		if guess(mid) == 0 {
			return mid
		}
		if guess(mid) < 0 {
			R = mid - 1
		} else {
			L = mid + 1
		}
	}
	return -1
}

/*
===================== 4、搜索旋转排序数组 =====================
整数数组 nums 按升序排列，数组中的值 互不相同 。
在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了
旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ...,
nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后
可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target，
则返回它的下标，否则返回 -1 。

示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例 2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例 3：
输入：nums = [1], target = 0
输出：-1

提示：
    1 <= nums.length <= 5000
    -10^4 <= nums[i] <= 10^4
    nums 中的每个值都 独一无二
    题目数据保证 nums 在预先未知的某个下标上进行了旋转
    -10^4 <= target <= 10^4

进阶：你可以设计一个时间复杂度为 O(log n) 的解决方案吗？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array
*/
/*
	方法一：二分查找
    算法与思路：
        虽然数组本身不是有序的，但是我们发现从中间分割数组的时候，，一定有一部分的数组是有序的。
        拿示例来看，我们从 6 这个位置分开以后数组变成了 [4, 5, 6] 和 [7, 0, 1, 2] 两个部分，
        其中左边 [4, 5, 6] 这个部分的数组是有序的，其他也是如此。

        这启示我们可以在常规二分搜索的时候查看当前 mid 为分割位置分割出来的两个部分 [l, mid]
        和 [mid + 1, r] 哪个部分是有序的，并根据有序的那个部分确定我们该如何改变二分搜索的上下界，
        因为我们能够根据有序的那部分判断出 target 在不在这个部分：
            如果 [l, mid - 1] 是有序数组，且 target 的大小满足 [nums[l],nums[mid])，
                则我们应该将搜索范围缩小至 [l, mid - 1]，否则在 [mid + 1, r] 中寻找。
            如果 [mid, r] 是有序数组，且 target 的大小满足 (nums[mid+1],nums[r]]，
                则我们应该将搜索范围缩小至 [mid + 1, r]，否则在 [l, mid - 1] 中寻找。
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func search(nums []int, target int) int {
	ln := len(nums)
	if ln == 0 {
		return -1
	}
	if ln == 1 {
		if nums[0] == target {
			return 0
		} else {
			return -1
		}
	}
	L, R := 0, ln-1
	for L <= R {
		mid := L + (R-L)>>1
		if nums[mid] == target {
			return mid
		}
		// 判断左边是否有序，左边有序则选择左边，否则选择右边
		if nums[0] <= nums[mid] {
			// 左边有序时，看看目标是否在左边，在则选择左边，否则选择右边
			if nums[0] <= target && target < nums[mid] {
				R = mid - 1
			} else {
				L = mid + 1
			}
		} else {
			// 左边无序时右边一定是有序的
			// 右边有序时，看看目标是否在右边，在则选择右边，否则选择左边
			if nums[mid] < target && target <= nums[ln-1] {
				L = mid + 1
			} else {
				R = mid - 1
			}
		}
	}
	return -1
}

/*
	方法二：使用模板三进行优化
*/
func search(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n-1
	for L+1 < R {
		mid := L + (R-L)>>1
		// 左边有序
		if nums[L] <= nums[mid] {
			if nums[L] <= target && target < nums[mid] {
				// 目标值在左边
				R = mid
			} else {
				L = mid
			}
		} else {
			// 右边有序
			if nums[mid] < target && target <= nums[R] {
				// 目标值在右边
				L = mid
			} else {
				R = mid
			}
		}
	}
	if nums[L] == target {
		return L
	}
	if nums[R] == target {
		return R
	}
	return -1
}

/* 
===================== 5、第一个错误的版本 =====================
你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有
通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本
都是错的。
假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元
测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

示例:
给定 n = 5，并且 version = 4 是第一个错误的版本。

调用 isBadVersion(3) -> false
调用 isBadVersion(5) -> true
调用 isBadVersion(4) -> true

所以，4 是第一个错误的版本。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xepthr/
*/
/* 
	方法一：二分查找【模板三】
	思路：
		以 [1, n] 为上下限调用 isBadVersion() 进行二分查找
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func firstBadVersion(n int) int {
	L, R := 1, n
	for L + 1 < R {
		mid := L + ((R - L) >> 1)
		if isBadVersion(mid) {
			R = mid
		} else {
			L = mid
		}
	}
	// 先处理 L，因为要返回第一个错误版本
	if isBadVersion(L) {
		return L
	}
	if isBadVersion(R) {
		return R
	}
	return -1
}

/* 
===================== 6、寻找峰值 =====================
峰值元素是指其值大于左右相邻值的元素。
给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情
况下，返回 任何一个峰值 所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞ 。

示例 1：
输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。

示例 2：
输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5 
解释：你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。

提示：
    1 <= nums.length <= 1000
    -2^31 <= nums[i] <= 2^31 - 1
    对于所有有效的 i 都有 nums[i] != nums[i + 1]

进阶：你可以实现时间复杂度为 O(logN) 的解决方案吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xem7js/
*/
/* 
	方法一：单调递增栈
	思路：
		利用单调递增栈的性质来寻找第一个峰值。
	时间复杂度：O(n)
		n 是数组元素个数，最坏情况下最后一个元素才是峰值，此时的时间复杂度
		为 O(n)
	空间复杂度：O(n)
		我们需要一个栈来存储元素，最坏情况下需要存储 n 个元素，此时空间复杂度
		为 O(n)
*/
type Node struct {
	index int
	value int
}
func findPeakElement(nums []int) int {
	n := len(nums))
	if n == 0 {
		return -1
	}
	stack := make([]*Node, 0)
	for i, v := range nums {
		node := &Node{index : i, value: v}
		if len(stack) == 0 {
			stack = append(stack, node)
			continue
		}
		// 查看栈顶元素
		top := stack[len(stack) - 1]
		// 如果当前元素小于栈顶元素，说明栈顶元素为峰值
		if node.value < top.value {
			return top.index
		}
		stack = append(stack, node)
	}
	return n - 1
}
/* 
	方法二：线性查找
	思路：
		利用 nums[i] != nums[i + 1] 的性质来判断 nums[i] 是否大于 nums[i+1]，
		如果大于，则 nums[i] 是峰值。
	时间复杂度：O(n)
		 我们对长度为 n 的数组 nums 只进行一次遍历
	空间复杂度：O(1)
*/
func findPeakElement(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	for i := 0; i < n - 1; i ++ {
		if nums[i] > nums[i + 1] {
			return i
		}
	}
	return n - 1
}

/* 
	方法三：二分查找【模板三】
	思路：
		我们可以将 nums 数组中的任何给定序列视为交替的升序和降序序列。通过利用这
		一点，以及“可以返回任何一个峰作为结果”的要求，我们可以利用二分查找来找到所
		需的峰值元素。
		
		在简单的二分查找中，我们处理的是一个有序数列，并通过在每一步减少搜索空间
		来找到所需要的数字。在本例中，我们对二分查找进行一点修改。首先从数组 nums
		中找到中间的元素 mid。若该元素恰好位于降序序列或者一个局部下降坡度中
		（通过将 nums[i] 与右侧比较判断)，则说明峰值会在本元素的左边。于是，我们
		将搜索空间缩小为 mid 的左边(包括其本身)，并在左侧子数组上重复上述过程。

		若该元素恰好位于升序序列或者一个局部上升坡度中（通过将 nums[i] 与右侧比较
		判断)，则说明峰值会在本元素的右边。于是，我们将搜索空间缩小为 mid 的右边，
		并在右侧子数组上重复上述过程。

		就这样，我们不断地缩小搜索空间，直到搜索空间中只有一个元素，该元素即为峰
		值元素。
	时间复杂度：O(logn)
		n 是数组元素个数，我们每一次对比都会将搜索空间减半。
	空间复杂度：O(1)
*/
func findPeakElement(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + ((R - L) >> 1)
		// nums[mid] 处于上升坡度，峰值在右边，往右找
		if nums[mid] < nums[mid + 1] {
			L = mid
		} else {
			R = mid
		}
	}
	// 在剩余的两个元素中寻找峰值
	if nums[L] > nums[R] {
		return L
	}
	return R
}

/* 
===================== 7、寻找旋转排序数组中的最小值 =====================
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。
例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
    若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
    若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]

注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], 
a[0], a[1], a[2], ..., a[n-2]] 。
给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形
进行了多次旋转。请你找出并返回数组中的 最小元素 。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。

示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。

示例 3：
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。

提示：
    n == nums.length
    1 <= n <= 5000
    -5000 <= nums[i] <= 5000
    nums 中的所有整数 互不相同
    nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xeawbd/
*/
/* 
	方法一：二分查找【模板三】
	思路：
		虽然旋转数组整体不是有序的，但是经过二分之后，[L, mid] 和 [mid, R] 总
		有一半是有序的，我们可以利用这个部分有序来找到最小值。
		在每一次二分对比过程中，我们可以对比 nums[mid] 和 nums[R]，如果：
			nums[mid] > nums[R]，说明 mid 左边有序，但最小值在 mid 右边，
			需要往右找，否则往左找
	时间复杂度：O(logn)
		n 是数组元素个数。
	空间复杂度：O(1)
*/
func findMin(nums []int) int {
	n := len(nums)
	L, R := 0, n - 1
	for L + 1 < R {
		mid :=  L + ((R - L) >> 1)
		if nums[mid] > nums[R] {
			// 左边有序，但最小值在右边，往右找
			L = mid
		} else {
			R = mid
		}
	}
	if nums[L] < nums[R] {
		return nums[L]
	}
	return nums[R]
}

/* 
===================== 8、在排序数组中查找元素的第一个和最后一个位置 =====================
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组
中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：
    你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？

示例 1：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

示例 2：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

示例 3：
输入：nums = [], target = 0
输出：[-1,-1]

提示：
    0 <= nums.length <= 10^5
    -10^9 <= nums[i] <= 10^9
    nums 是一个非递减数组
    -10^9 <= target <= 10^9

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xenp13/
*/
/* 
	方法一：二分查找【库函数】
	思路：
		因为数组是有序的，所以我们可以用二分查找。
		对于起点，我们需要从数组中找到大于等于 target 的第一个元素的下标。
		对于终点，我们需要从数组中找到大于 target 的第一个元素的下标，再减1
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
// SearchInts 在递增顺序的a中搜索x，返回x的索引。如果查找不到，
// 返回值是x应该插入a的位置（以保证a的递增顺序），返回值可以是len(a)。
func searchRange(nums []int, target int) []int {
	// SearchInts 函数内部实现也是使用二分查找【模板二】
    leftmost := sort.SearchInts(nums, target)
    if leftmost == len(nums) || nums[leftmost] != target {
        return []int{-1, -1}
    }
    rightmost := sort.SearchInts(nums, target + 1) - 1
    return []int{leftmost, rightmost}
}

/* 
	方法二：二分查找【模板二】
	思路：
		因为数组是有序的，所以我们可以用二分查找。
		对于起点，我们需要从数组中找到大于等于 target 的第一个元素的下标。
		对于终点，我们需要从数组中找到大于 target 的第一个元素的下标，再减1
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func searchRange(nums []int, target int) []int {
	// 返回目标值第一次出现的下标，找不到则返回它该插入的位置，返回值可能是 len(nums)
	var search func(nums []int, target int) int 
	search = func(nums []int, target int) int {
		n := len(nums)
		if n == 0 {
			return -1
		}
		L, R := 0, n
		for L < R {
			mid := L + ((R - L) >> 1)
			// 小于等于往左找，找到目标值第一个出现的下标
			if target <= nums[mid] {
				R = mid
			} else {
				L = mid + 1
			}
		}
		// 如果找不到，则返回该值该插入的位置，返回可能是 n
		return L
	}
	start := search(nums, target)
	if (start >= len(nums) || start < 0) || nums[start] != target {
		return []int{-1, -1}
	}
	end := search(nums, target + 1) - 1
	return []int{start, end}
}

/* 
===================== 9、找到 K 个最接近的元素 =====================
给定一个排序好的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）
的 k 个数。返回的结果必须要是按升序排好的。
整数 a 比整数 b 更接近 x 需要满足：
    |a - x| < |b - x| 或者
    |a - x| == |b - x| 且 a < b

示例 1：
输入：arr = [1,2,3,4,5], k = 4, x = 3
输出：[1,2,3,4]

示例 2：
输入：arr = [1,2,3,4,5], k = 4, x = -1
输出：[1,2,3,4]

提示：
    1 <= k <= arr.length
    1 <= arr.length <= 10^4
    数组里的每个元素与 x 的绝对值不超过 10^4

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xeve4m/
*/
/* 
	方法一：二分查找【模板二】
	思路：
		使用二分查找，把左右边界界定为：L, R = 0, n - k，然后寻找左边界返回即可，
		二分条件：
			如果 x 与左边界点差值 > 右边界点与 x 的差值向右找，否则向左找
	时间复杂度：O(logn)
		n 是数组元素个数，我们只进行一次二分查找，耗时 O(logn)
	空间复杂度：O(1)
*/
func findClosestElements(arr []int, k int, x int) []int {
	n := len(arr)
	if n == 0 {
		return []int{}
	}
	// 如此划分右边界，则退出循环时，L 最大也只是 n - k
	L, R := 0, n - k
	for L < R {
		mid := L + ((R - L) >> 1)
		// 尝试以 arr[mid] 作为左边界，则右边界为 arr[mid + k]
		// 如果 x 与左边界点的差值 > 右边界点与 x 的差值
		// 说明 x 本身在 mid 的右边，即离 x 更近的 k 个数在 mid 右边，向右找
		// 否则向左找
		if x - arr[mid] > arr[mid + k] - x {
			L = mid + 1
		} else {
			R = mid
		}
	}
	// 无需排序，因为原数组是有序的
	return arr[L : L + k]
}

/* 
===================== 10、寻找峰值 =====================
峰值元素是指其值大于左右相邻值的元素。
给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情
况下，返回 任何一个峰值 所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞ 。

示例 1：
输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。

示例 2：
输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5 
解释：你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。

提示：
    1 <= nums.length <= 1000
    -2^31 <= nums[i] <= 2^31 - 1
    对于所有有效的 i 都有 nums[i] != nums[i + 1]

进阶：你可以实现时间复杂度为 O(logN) 的解决方案吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xem7js/
*/
/* 
	方法一：单调递增栈
	思路：
		利用单调递增栈的性质来寻找第一个峰值。
	时间复杂度：O(n)
		n 是数组元素个数，最坏情况下最后一个元素才是峰值，此时的时间复杂度
		为 O(n)
	空间复杂度：O(n)
		我们需要一个栈来存储元素，最坏情况下需要存储 n 个元素，此时空间复杂度
		为 O(n)
*/
type Node struct {
	index int
	value int
}
func findPeakElement(nums []int) int {
	n := len(nums))
	if n == 0 {
		return -1
	}
	stack := make([]*Node, 0)
	for i, v := range nums {
		node := &Node{index : i, value: v}
		if len(stack) == 0 {
			stack = append(stack, node)
			continue
		}
		// 查看栈顶元素
		top := stack[len(stack) - 1]
		// 如果当前元素小于栈顶元素，说明栈顶元素为峰值
		if node.value < top.value {
			return top.index
		}
		stack = append(stack, node)
	}
	return n - 1
}
/* 
	方法二：线性查找
	思路：
		利用 nums[i] != nums[i + 1] 的性质来判断 nums[i] 是否大于 nums[i+1]，
		如果大于，则 nums[i] 是峰值。
	时间复杂度：O(n)
		 我们对长度为 n 的数组 nums 只进行一次遍历
	空间复杂度：O(1)
*/
func findPeakElement(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	for i := 0; i < n - 1; i ++ {
		if nums[i] > nums[i + 1] {
			return i
		}
	}
	return n - 1
}

/* 
	方法三：二分查找【模板三】
	思路：
		我们可以将 nums 数组中的任何给定序列视为交替的升序和降序序列。通过利用这
		一点，以及“可以返回任何一个峰作为结果”的要求，我们可以利用二分查找来找到所
		需的峰值元素。
		
		在简单的二分查找中，我们处理的是一个有序数列，并通过在每一步减少搜索空间
		来找到所需要的数字。在本例中，我们对二分查找进行一点修改。首先从数组 nums
		中找到中间的元素 mid。若该元素恰好位于降序序列或者一个局部下降坡度中
		（通过将 nums[i] 与右侧比较判断)，则说明峰值会在本元素的左边。于是，我们
		将搜索空间缩小为 mid 的左边(包括其本身)，并在左侧子数组上重复上述过程。

		若该元素恰好位于升序序列或者一个局部上升坡度中（通过将 nums[i] 与右侧比较
		判断)，则说明峰值会在本元素的右边。于是，我们将搜索空间缩小为 mid 的右边，
		并在右侧子数组上重复上述过程。

		就这样，我们不断地缩小搜索空间，直到搜索空间中只有一个元素，该元素即为峰
		值元素。
	时间复杂度：O(logn)
		n 是数组元素个数，我们每一次对比都会将搜索空间减半。
	空间复杂度：O(1)
*/
func findPeakElement(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + ((R - L) >> 1)
		// nums[mid] 处于上升坡度，峰值在右边，往右找
		if nums[mid] < nums[mid + 1] {
			L = mid
		} else {
			R = mid
		}
	}
	// 在剩余的两个元素中寻找峰值
	if nums[L] > nums[R] {
		return L
	}
	return R
}

/* 
===================== 11、寻找旋转排序数组中的最小值 II =====================
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。
例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
    若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
    若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]

注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0],
a[1], a[2], ..., a[n-2]] 。
给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情
形进行了多次旋转。请你找出并返回数组中的 最小元素 。

示例 1：
输入：nums = [1,3,5]
输出：1

示例 2：
输入：nums = [2,2,2,0,1]
输出：0

提示：
    n == nums.length
    1 <= n <= 5000
    -5000 <= nums[i] <= 5000
    nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转

进阶：
    这道题是 寻找旋转排序数组中的最小值 的延伸题目。
    允许重复会影响算法的时间复杂度吗？会如何影响，为什么？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xetw7g/
*/
/* 
    方法一：二分查找【模板三】
    思路：
        即便有旋转和重复元素，我们依旧可以按照之前的办法来处理，只需要在每一次二分
        过程中跳过重复元素即可。
        在每一次二分过程中，对比 nums[mid] 和 nums[R]：
            如果 nums[mid] > nums[R]，说明数组被旋转，且 mid 的左边有序，
            但最小值在右边，需要往右找，否则往左找。
    时间复杂度：O(n)
        其中 n 是数组 nums 的长度。如果数组是随机生成的，那么数组中包含相同元素
        的概率很低，在二分查找的过程中，大部分情况都会忽略一半的区间，此时时间复杂
        度为 O(logn)。而在最坏情况下，如果数组中的元素完全相同，过滤重复元素需要
        执行 n 次，此时时间复杂度为 O(n)。
    空间复杂度：O(1)
*/
func findMin(nums []int) int {
    n := len(nums)
    L, R := 0, n - 1
    for L + 1 < R {
        // 跳过重复元素
        for L < R && nums[R] == nums[R - 1] {
            R --
        }
        for L < R && nums[L] == nums[L + 1] {
            L ++
        }
        mid := L + ((R - L) >> 1)
        // nums[mid] > nums[R]，说明 mid 左边有序，但最小值在 
        // mid 右边，往右找，否则往左找
        if nums[mid] > nums[R] {
            L = mid
        } else {
            R = mid
        }
    }
    if nums[L] < nums[R] {
        return nums[L]
    }
    return nums[R]
}

/* 
===================== 12、两个数组的交集 =====================
给定两个数组，编写一个函数来计算它们的交集。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例 2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]

说明：
    输出结果中的每个元素一定是唯一的。
    我们可以不考虑输出结果的顺序。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe820v/
*/
/* 
	方法一：哈希表法
	思路：
		先用哈希表记录长度较短的数组的元素，然后遍历另一个数组，如果
		该数组元素在哈希表中出现过，则说明它是交集元素。
	时间复杂度：O(m + n)
		m、n 分别是两个数组的元素个数，我们需要遍历两个数组进行处理。
	空间复杂度：O(min(m, n))
		m、n 分别是两个数组的元素个数，我们需要记录较短的那个数组的元素。
*/
func intersection(nums1 []int, nums2 []int) []int {
	// 始终保证 nums2 是较长数组
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	hash := make(map[int]bool, 0)
	for _, v := range nums1 {
		hash[v] = true
	}
	// 交集需要去重
	resultMap := make(map[int]bool, 0)
	for _, v := range nums2 {
		if _, ok := hash[v]; ok {
			resultMap[v] = true
		}
	}
	ans := make([]int, 0)
	for k, _ := range resultMap {
		ans = append(ans, k)
	}
	return ans
}

/* 
	方法二：排序+双指针
	思路：
		如果两个数组是有序的，则可以使用双指针的方法得到两个数组的交集。
		首先对两个数组进行排序，然后使用两个指针遍历两个数组。可以预见的
		是加入答案的数组的元素一定是递增的，为了保证加入元素的唯一性，我
		们需要额外记录变量 pre 表示上一次加入答案数组的元素。
		初始时，两个指针分别指向两个数组的头部。每次比较两个指针指向的两
		个数组中的数字，如果两个数字不相等，则将指向较小数字的指针右移一
		位，如果两个数字相等，且该数字不等于 pre，将该数字添加到答案并更
		新 pre 变量，同时将两个指针都右移一位。当至少有一个指针超出数组
		范围时，遍历结束。
	时间复杂度：O(mlog⁡m+nlog⁡n)
		其中 m 和 n 分别是两个数组的长度。对两个数组排序的时间复杂度分
		别是 O(mlog⁡m) 和 O(nlog⁡n)，双指针寻找交集元素的时间复杂度
		是 O(m+n)，因此总时间复杂度是 O(mlog⁡m+nlog⁡n)。
	空间复杂度：O(log⁡m+log⁡n)
		其中 m 和 n 分别是两个数组的长度。空间复杂度主要取决于排序使用的
		额外空间。
*/
func intersection(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	m, n := len(nums1), len(nums2)
	i, j := 0, 0
	ans := make([]int, 0)
	for i < m && j < n {
		if nums1[i] == nums2[j] {
			// 答案中没有的数才添加进答案
			if len(ans) == 0 || nums1[i] != ans[len(ans) - 1] {
				ans = append(ans, nums1[i])
			}
			i ++
			j ++
			continue
		}
		if nums1[i] < nums2[j] {
			i ++
		} else {
			j ++
		}
	}
	return ans
}
/* 
	方法三：排序 + 二分查找
	思路：
		我们选取较短数组的不重复元素作为二分查找的key，对较长数组进行排序，然后用
		key 在排序数组中进行查找，找到说明该 key 是交集元素，否则不是交集元素。
	时间复杂度：O(max(m,n)*log(max(m,n)))
		m、n 分别是 nums1、nums2 的长度。
		我们先遍历较短数组得到不重复key，耗时O(min(m,n))，再对较长数组进行排序，
		耗时 O(max(m,n)*log(max(m,n))))，接着用不重复 key 到排序数组中进行二分
		查找，耗时 O(min(m,n)*log(max(m,n)))，最后遍历交集元素进行返回，耗时 
		O(min(m,n))，所以总的时间复杂度是 O(max(m,n)log(max(m,n)))。
	空间复杂度：O(min(m,n) + log(max(m,n)))
		m、n 分别是 nums1、nums2 的长度。
		我们需要存储较短数组的不重复元素作为二分查找的 key，最坏情况下较短数组的
		元素都不重复，此时需要 O(min(m,n)) 的额外空间，对较长数组进行排序时需要
		O(log(max(m,n))) 的额外空间，所以总的空间复杂度是 O(min(m,n) + log(max(m,n)))
*/
func intersection(nums1 []int, nums2 []int) []int {
	// 始终保证 nums2 是较长数组
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	if len(nums1) == 0 || len(nums2) == 0 {
		return []int{}
	}
	keyMap := make(map[int]bool, 0)
	for _, v := range nums1 {
		keyMap[v] = false
	}
	sort.Ints(nums2)
	for k, _ := range keyMap {
		keyMap[k] = binarySearch(nums2, k)
	}
	ans := make([]int, 0)
	for k, v := range keyMap {
		if v {
			ans = append(ans, k)
		}
	}
	return ans
}
func binarySearch(nums []int, target int) bool {
	n := len(nums)
	if n == 0 {
		return false
	}
	L, R := 0, n
	for L < R {
		mid := L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	if L < n && nums[L] == target {
		return true
	}
	return false
}

/* 
===================== 13、两个数组的交集 II =====================
给定两个数组，编写一个函数来计算它们的交集。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

说明：
    输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
    我们可以不考虑输出结果的顺序。
进阶：
    如果给定的数组已经排好序呢？你将如何优化你的算法？
    如果 nums1 的大小比 nums2 小很多，哪种方法更优？
	如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到
	内存中，你该怎么办？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe3pwj/
*/
/* 
	方法一：哈希表法
	思路：
		采用哈希表记录较短数组的元素及其出现次数，记为 hash1，再以另一个哈希表记录
		较长数组的元素及其出现次数，记为 hash2，最后遍历 hash2 的元素，如果该元素
		在 hash1 中出现过，则 k 是交集元素，交集次数为 min(hash1[k], hash2[k])。
	时间复杂度：O(m + n)
	空间复杂度：O(m + n)
*/
func intersect(nums1 []int, nums2 []int) []int {
	// 始终保证 nums2 为较长数组
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	hash1 := make(map[int]int, 0)
	for _, v := range nums1 {
		hash1[v] ++
	}
	hash2 := make(map[int]int, 0)
	for _, v := range nums2 {
		hash2[v] ++
	}
	ans := make([]int, 0)
	for k, v2 := range hash2 {
		if v1, ok := hash1[k]; ok {
			// 按出现次数把交集元素放入结果集
			for i := 0; i < min(v1, v2); i ++ {
				ans = append(ans, k)
			}
		}
	}
	return ans
}
func min(a, b int) int{
	if a < b {
		return a
	}
	return b
}

/* 
	方法二：哈希表法（空间优化）
	思路：
		我们可以只用一个哈希表来完成。
		我们用一个哈希表记录较短数组元素出现的次数，然后遍历另一个数组，
		如果该数组元素在哈希表中出现过，则我们把它添加到结果集一次，并
		在哈希表中减少一次该元素出现的次数。
	时间复杂度：O(m + n)
	空间复杂度：O(min(m, n))
*/
func intersect(nums1 []int, nums2 []int) []int {
	hash := make(map[int]int, 0)
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	for _, v := range nums1 {
		hash[v] ++
	}
	res := make([]int, 0)
	for _, v := range nums2 {
		if hash[v] > 0 {
			res = append(res, v)
			hash[v] --
		}
	}
	return res
}

/* 
	方法三：排序+双指针
	思路：
		排序两个数组，然后用两个指针 i, j 同时遍历两个数组，如果 
			nums1[i] < nums2[j], i ++
			nums1[i] > nums2[j], j ++
			nums1[i] == nums2[j]，i ++, j ++ 并把元素添加到结果集
	时间复杂度：O(mlogm + nlogn)
		其中 m 和 n 分别是两个数组的长度。对两个数组进行排序的时间复杂度
		是 O(mlog⁡m + nlog⁡n)，遍历两个数组的时间复杂度是 O(m + n)，
		因此总时间复杂度是 O(mlog⁡m + nlog⁡n)。
	空间复杂度：O(min⁡(m,n))
		其中 m 和 n 分别是两个数组的长度。为返回值创建一个数组 
		intersection，其长度为较短的数组的长度。
	
	结语
		如果 nums2​ 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次
		加载所有的元素到内存中。那么就无法高效地对 nums2​ 进行排序，因此
		推荐使用方法二而不是方法三。在方法二中，nums2​ 只关系到查询操作，
		因此可以每次只读取 nums2​ 中的一部分数据，并进行处理即可。
*/
func intersect(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	res := make([]int, 0)
	i, j := 0, 0
	for i < len(nums1) && j < len(nums2) {
		if nums1[i] == nums2[j] {
			res = append(res, nums1[i])
			i ++
			j ++
			continue
		}
		if nums1[i] < nums2[j] {
			i ++
		} else {
			j ++
		}
	}
	return res
}

/*
===================== 14、两数之和 II - 输入有序数组 =====================
给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之
和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开
始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。

你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例 1：
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

示例 2：
输入：numbers = [2,3,4], target = 6
输出：[1,3]

示例 3：
输入：numbers = [-1,0], target = -1
输出：[1,2]

提示：
    2 <= numbers.length <= 3 * 10^4
    -1000 <= numbers[i] <= 1000
    numbers 按 递增顺序 排列
    -1000 <= target <= 1000
    仅存在一个有效答案

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xeqevt/
*/
/* 
	方法一：二分查找【模板二】
	思路：
		先以 key = target - nums[i] 的方式固定住一个数，再用二分查找的方式从
		nums[i+1, n) 中查找 key，n 是数组长度，如果找到 key，则返回。
		注意，返回结果的下标需要从 1 开始。
	时间复杂度：O(n)
		n 是数组长度，我们需要先遍历 nums 来固定一个数 nums[i]，再从剩余数组中
		对另一个数进行二分查找，最坏情况下两个符合条件的数在数组的最后面，此时
		时间复杂度为 O(n)。
	空间复杂度：O(1)
*/
func twoSum(numbers []int, target int) []int {
	n := len(numbers)
	for i := 0; i < n; i ++ {
		key := target - numbers[i]
		idx := binarySearch(numbers, key, i + 1, n)
		if idx != -1 {
			return []int{i + 1, idx + 1}
		}
	}
	return []int{}
}
func binarySearch(nums []int, target, L, R int) int {
	for L < R {
		mid := L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	if L < len(nums) && nums[L] == target {
		return L
	}
	return -1
}
/* 
	方法二：双指针
	思路：
		初始时两个指针分别指向第一个元素位置和最后一个元素的位置。
		每次计算两个指针指向的两个元素之和，并和目标值比较。
		如果两个元素之和等于目标值，则发现了唯一解。如果两个元素之和小于目标值，
		则将左侧指针右移一位。如果两个元素之和大于目标值，则将右侧指针左移一位。
		移动指针之后，重复上述操作，直到找到答案。
	时间复杂度：O(n)
		其中 n 是数组的长度。两个指针移动的总次数最多为 n 次。
    空间复杂度：O(1)
*/
func twoSum(numbers []int, target int) []int {
	n := len(numbers)
	if n < 2 {
		return []int{-1, -1}
	}
	L, R := 0, n - 1
	for L < R {
		sum := numbers[L] + numbers[R]
		if sum == target {
			return []int{L + 1, R + 1}
		}
		if sum < target {
			L ++
		} else {
			R --
		}
	}
	return []int{-1, -1}
}