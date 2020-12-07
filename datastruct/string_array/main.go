package main

import (
	"fmt"
	"sort"
	"strings"
)

/* 
====================== 1、寻找数组的中心索引 =========================
给定一个整数类型的数组 nums，请编写一个能够返回数组 “中心索引” 的方法。
我们是这样定义数组 中心索引 的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1：
输入：
nums = [1, 7, 3, 6, 5, 6]
输出：3
解释：
索引 3 (nums[3] = 6) 的左侧数之和 (1 + 7 + 3 = 11)，与右侧数之和 (5 + 6 = 11) 相等。
同时, 3 也是第一个符合要求的中心索引。

示例 2：
输入：
nums = [1, 2, 3]
输出：-1
解释：
数组中不存在满足此条件的中心索引。

说明：
    nums 的长度范围为 [0, 10000]。
    任何一个 nums[i] 将会是一个范围在 [-1000, 1000]的整数。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/find-pivot-index
*/
/* 
	算法：前缀和
	思路：
		S 是数组的和，当索引 i 是中心索引时，位于 i 左边数组元素的和 leftsum 满足 S - nums[i] - leftsum。
		我们只需要判断当前索引 i 是否满足 leftsum==S-nums[i]-leftsum 并动态计算 leftsum 的值。
	时间复杂度：O(n)
		n 表示数组长度。
	空间复杂度：O(1)
*/
func pivotIndex(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	sum := 0
	for i := 0; i < n; i ++ {
		sum += nums[i]
	}
	leftSum := 0
	for i := 0; i < n; i ++ {
		// 不计算当前位置的值
		if leftSum == sum - nums[i] - leftSum {
			return i
		}
		leftSum += nums[i]
	}
	return -1
}

/* 
====================== 2、搜索插入位置 =========================
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

示例 1:
输入: [1,3,5,6], 5
输出: 2

示例 2:
输入: [1,3,5,6], 2
输出: 1

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cxqdh/
*/
/* 
	算法：二分查找
	思路：
		利用二分查找模板三进行寻找，当 target <= nums[L] 时插入到 L 位置，
		当 nums[L] < target <= nums[L] 时插入到 R 位置，否则插入到 R + 1 位置。
	时间复杂度：O(logn)
		n 表示数组长度。
	空间复杂度：O(1)
*/
func searchInsert(nums []int, target int) int {
	n := len(nums)
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + (R - L) >> 1
		// 等于往左边找
		if nums[mid] >= target {
			R = mid
		} else {
			L = mid
		}
	}
	if target <= nums[L] {
		return L
	} else if target <= nums[R] {
		return R
	} else {
		return R + 1
	}
}

/* 
====================== 3、合并区间 =========================
给出一个区间的集合，请合并所有重叠的区间。

示例 1:
输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2:
输入: intervals = [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

注意：输入类型已于2019年4月15日更改。 请重置默认代码定义以获取新方法签名。

提示：
    intervals[i][0] <= intervals[i][1]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c5tv3/
*/

/* 
	算法：排序合并
	思路：
		首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间
		加入 merged 数组中，并按顺序依次考虑之后的每个区间：
			如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，
				那么它们不会重合，我们可以直接将这个区间加入数组 merged 的末尾；
			否则，它们重合，我们需要用当前区间的右端点更新数组 merged 
				中最后一个区间的右端点，将其置为二者的较大值。
	时间复杂度：O(nlog⁡n)
		其中 n 为区间的数量。除去排序的开销，我们只需要一次线性扫描，
		所以主要的时间开销是排序的 O(nlogn)。
	空间复杂度：O(log⁡n)
		其中 n 为区间的数量。这里计算的是存储答案之外，
		使用的额外空间。O(log⁡n) 即为排序所需要的空间复杂度。
*/
func merge(intervals [][]int) [][]int {
	res := [][]int{}
	n := len(intervals)
	if n == 0 {
		return res
	}
	sort.Sort(intss(intervals))
	for i := 0; i < n; i ++ {
		L, R := intervals[i][0], intervals[i][1]
		// 如果结果集为空 或 当前区间与上一区间不重叠，则直接存放
		if len(res) == 0 || L > res[len(res) - 1][1] {
			res = append(res, []int{L, R})
		} else {
			// 合并区间
			res[len(res) - 1][1] = max(res[len(res) - 1][1], R)
		}
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
// 自定义排序
type intss [][]int
func (this intss) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}
func (this intss) Len() int {
	return len(this)
}
func (this intss) Less(i, j int) bool {
	return this[i][0] < this[j][0]
}

/* 
====================== 4、旋转矩阵 =========================
给你一幅由 N × N 矩阵表示的图像，其中每个像素的大小为 4 字节。请你设计一种算法，将图像旋转 90 度。
不占用额外内存空间能否做到？

示例 1:
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/clpgd/
*/

/* 
方法二：原地旋转
	我们已知旋转公式：
		（1） matrix[col][n-r-1] = matrix[r][c]
	此时如对 matrix[r][c] 进行旋转则会导致 matrix[col][n-r-1] 被覆盖，需要一个临时变量来保存它。
	那么 matrix[col][n-r-1] 旋转后应该被放入哪里呢？
		由（1）公式推导得：
		（2）matrix[n-r-1][n-col-1] = matrix[col][n-r-1]
	同理 matrix[n-r-1][n-col-1] 旋转后应该放入哪里？
		由（1）公式推导得：
		（3）matrix[n-col-1][r] = matrix[n-r-1][n-col-1]
	同理 matrix[n-col-1][r] 旋转后该放入哪里？
		由（1）公式推导得：
		（4）matrix[r][c] = matrix[n-col-1][r]
	由（1）（2）（3）（4）可知这是一个循环，故我们只需要一个临时变量就可以实现原地交换了：
		tmp := matrix[r][c]
		matrix[r][c] = matrix[n-col-1][r]
		matrix[n-col-1][r] = matrix[n-r-1][n-col-1]
		matrix[n-r-1][n-col-1] = matrix[col][n-r-1]
		matrix[col][n-r-1] = tmp
	当我们知道了如何原地旋转矩阵之后，还有一个重要的问题在于：
		我们应该枚举哪些位置 (row,col)进行上述的原地交换操作呢？
	由于每一次原地交换四个位置，因此：
		当 n 为偶数时，我们需要枚举 n^2/4=(n/2)∗(n/2) 个位置，矩阵的左上角符合我们的要求。
		例如当 n=4 时，下面第一个矩阵中 ∗ 所在就是我们需要枚举的位置，
		每一个 ∗ 完成了矩阵中四个不同位置的交换操作：
			**..              ..**              ....              ....
			**..   =下一项=>   ..**   =下一项=>   ....   =下一项=>   ....
			....              ....              ..**              **..
			....              ....              ..**              **..
		保证了不重复、不遗漏；
		当 n 为奇数时，由于中心的位置经过旋转后位置不变，我们需要枚举 (n^2−1)/4=((n−1)/2)∗((n+1)/2) 个位置，
		同样可以使用矩阵左上角对应大小的子矩阵。例如当 n=5 时，下面第一个矩阵中 ∗ 所在就是我们需要枚举的位置，
		每一个 ∗ 完成了矩阵中四个不同位置的交换操作：
			***..              ...**              .....              .....
			***..              ...**              .....              .....
			..x..   =下一项=>   ..x**   =下一项=>   ..x..   =下一项=>   **x..
			.....              .....              ..***              **...
			.....              .....              ..***              **...
    	同样保证了不重复、不遗漏。
	综上所述，我们只需要枚举矩阵左上角高为 ⌊n/2⌋，宽为 ⌊(n+1)/2⌋ 的子矩阵即可。
	时间复杂度：O(N^2)
		其中 N 是 matrix 的边长。对于每一次翻转操作，我们都需要枚举矩阵中一半的元素。
	空间复杂度：O(1)
		为原地翻转得到的原地旋转。
*/
func rotate(matrix [][]int)  {
	rn := len(matrix)
	if rn == 0 {
		return
	}
	cn := len(matrix[0])
	if cn == 0 {
		return
	}
	for r := 0; r < rn >> 1; r ++ {
		for c := 0; c < (cn + 1) >> 1; c ++ {
			tmp := matrix[r][c]
			matrix[r][c] = matrix[rn - 1 - c][r]
			matrix[rn - 1 - c][r] = matrix[rn - 1 - r][rn - 1 - c]
			matrix[rn - 1 - r][rn - 1 - c] = matrix[c][rn - 1 - r]
			matrix[c][rn - 1 - r] = tmp
		}
	}
}

/* 
====================== 5、零矩阵 =========================
编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。

示例 1：
输入：
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出：
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

示例 2：
输入：
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出：
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/ciekh/
*/
func setZeroes(matrix [][]int)  {
    rn := len(matrix)
    if rn == 0 {
        return
    }
	cn := len(matrix[0])
	if cn == 0 {
		return
	}
    firstRowHasZero, firstColHasZero := false, false
    for c := 0; c < cn; c ++ {
        if matrix[0][c] == 0 {
            firstRowHasZero = true
            break
        }
    }
    for r := 0; r < rn; r ++ {
        if matrix[r][0] == 0 {
            firstColHasZero = true
            break
        }
    }
    for r := 1; r < rn; r ++ {
        for c := 1; c < cn; c ++ {
            // 把这个0在第一行、第一列上的投影设为0
            if matrix[r][c] == 0 {
                matrix[0][c] = 0
                matrix[r][0] = 0
            }
        }
    }
    for r := 1; r < rn; r ++ {
        for c := 1; c < cn; c ++ {
            // 如果该位置在第一行、第一列上的投影为0，则设为0 
            if matrix[0][c] == 0 || matrix[r][0] == 0 {
                matrix[r][c] = 0
            }
        }
    }
    // 最后再处理第一行、第一列
    if firstRowHasZero {
        for c := 0; c < cn; c ++ {
            matrix[0][c] = 0
        }
    }
    if firstColHasZero {
        for r := 0; r < rn; r ++ {
            matrix[r][0] = 0
        }
    }
}


/* 
====================== 6、对角线遍历 =========================
给定一个含有 M x N 个元素的矩阵（M 行，N 列），请以对角线遍历的顺序返回这个矩阵中的所有元素，对角线遍历如下图所示。

示例:
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]

输出:  [1,2,4,7,5,3,6,8,9]
说明:
    给定矩阵中的元素总数不会超过 100000 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/diagonal-traverse
*/
/* 
	思路：
		1、每一趟对角线中元素的坐标（x, y）相加的和是递增的。
				第一趟：1 的坐标(0, 0)。x + y == 0。
				第二趟：2 的坐标(1, 0)，4 的坐标(0, 1)。x + y == 1。
				第三趟：7 的坐标(0, 2), 5 的坐标(1, 1)，3 的坐标(2, 0)。第三趟 x + y == 2。
				第四趟：……
		2、每一趟都是 x 或 y 其中一个从大到小（每次-1），另一个从小到大（每次+1）。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x 每次-1，y 每次+1。
				第三趟：7 的坐标(2, 0) 5 的坐标(1, 1)，3 的坐标(0, 2)。x 每次 +1，y 每次 -1。
		3、确定初始值。例如这一趟是 x 从大到小， x 尽量取最大，当初始值超过 x 的上限时，
			不足的部分加到 y 上面。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x + y == 1，x 初始值取 1，y 取 0。
				第四趟：6 的坐标(1, 2)，8 的坐标(2, 1)。x + y == 3，x 初始值取 2，剩下的加到 y上，y 取 1。
		4、确定结束值。例如这一趟是 x 从大到小，这一趟结束的判断是， x 减到 0 或者 y 加到上限。
				第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x 减到 0 为止。
				第四趟：6 的坐标(1, 2)，8 的坐标(2, 1)。x 虽然才减到 1，但是 y 已经加到上限了。
		5、这一趟是 x 从大到小，那么下一趟是 y 从大到小，循环进行。 
			并且方向相反时，逻辑处理是一样的，除了x，y和他们各自的上限值是相反的。
				x 从大到小，第二趟：2 的坐标(0, 1)，4 的坐标(1, 0)。x + y == 1，
					x 初始值取 1，y 取 0。结束值 x 减到 0 为止。
				x 从小到大，第三趟：7 的坐标(2, 0)，5 的坐标(1, 1)，3 的坐标(0, 2)。
					x + y == 2，y 初始值取 2，x 取 0。结束值 y 减到 0 为止。
*/
func findDiagonalOrder(matrix [][]int) []int {
	res := make([]int, 0)
	m := len(matrix)
	if m == 0 {
		return res
	}
	n := len(matrix[0])
	if n == 0 {
		return res
	}

	// 标识方向，以确定上限的转换
	flag := true
	for i := 0; i < m + n - 1; i ++ {
		var pm, pn int
		if flag {
			pm, pn = m, n
		} else {
			pm, pn = n, m
		}

		var x, y int
		if i < pm {
			x = i
		} else {
			x = pm - 1
		}
		y = i - x
		for x >= 0 && y < pn {
			if flag {
				res = append(res, matrix[x][y])
			} else {
				res = append(res, matrix[y][x])
			}
			x --
			y ++
		}
		flag = !flag
	}
	return res
}

/* 
====================== 7、最长公共前缀 =========================
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例 1:
输入: ["flower","flow","flight"]
输出: "fl"

示例 2:
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。

说明:
所有输入只包含小写字母 a-z 。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/ceda1/
*/
/* 
	方法一：纵向对比
	思路：
		纵向扫描时，从前往后遍历所有字符串的每一列，比较相同列上的字符是否相同，
		如果相同则继续对下一列进行比较，如果不相同则当前列不再属于公共前缀，
		当前列之前的部分为最长公共前缀。
	时间复杂度：O(MN)
		M表示公共前缀的长度，N表示数组元素的个数，最坏情况下数组中所有元素都相同，
		需要对比每个元素中的每一个字符。
	空间复杂度：O(1)，我们只需要常数的额外空间
*/
func longestCommonPrefix(strs []string) string {
	n := len(strs)
	if n == 0 {
		return ""
	}
	if n == 1 {
		return strs[0]
	}
	ans := strs[0]
	var i int
	for i = 0; i < len(ans); i ++ {
		for j := 1; j < n; j ++ {
			if i >= len(strs[j]) || strs[j][i] != ans[i] {
				return ans[:i]
			}
		}
	}
	return ans
}
/* 
	方法二：分治法
	思路：
		最长公共前缀满足结合律：LCP(S1​…Sn​)=LCP(LCP(S1​…Sk​),LCP(Sk+1​…Sn​))
		其中 LCP(S1…Sn) 是字符串 S1…Sn​ 的最长公共前缀，1<k<n。
		基于上述结论，可以使用分治法得到字符串数组中的最长公共前缀。
		对于问题 LCP(Si⋯Sj)，可以分解成两个子问题 LCP(Si…Smid) 与 LCP(Smid+1…Sj)，
		其中 mid=(i+j)/2​。对两个子问题分别求解，然后对两个子问题的解计算最长公共前缀，即为原问题的解。
	时间复杂度：O(mn)
		其中 m 是字符串数组中的字符串的平均长度，n 是字符串的数量。
		时间复杂度的递推式是 T(n)=2⋅T(n/2)+O(m)，通过计算可得 T(n)=O(mn)。
	空间复杂度：O(mlog⁡n)
		其中 m 是字符串数组中的字符串的平均长度，n 是字符串的数量。
		空间复杂度主要取决于递归调用的层数，层数最大为 log⁡n，每层需要 m 的空间存储返回结果。
*/
func longestCommonPrefix2(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	var lcp func(int, int) string
	lcp = func(start, end int) string {
		// 开始与结束相等说明只有一个元素
		if start == end {
			return strs[start]
		}
		// divide
		mid := (start + end) / 2
		lcpLeft := lcp(start, mid)
		lcpRight := lcp(mid + 1, end)

		// conquer
		minLen := min(len(lcpLeft), len(lcpRight))
		for i := 0; i < minLen; i ++ {
			if lcpLeft[i] != lcpRight[i] {
				return lcpLeft[:i]
			}
		}
		return lcpLeft[:minLen]
	}
	return lcp(0, len(strs) - 1)
}

/* 
====================== 8、最长回文子串 =========================
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。

示例 2：
输入: "cbbd"
输出: "bb"

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/conm7/
*/
/* 
	方法一：动态规划
	思路与算法:
		对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。
		例如对于字符串 “ababa”，如果我们已经知道 “bab”是回文串，那么 “ababa” 一定是回文串，
		这是因为它的首尾两个字母都是 “a”。
		根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 P(i,j) 表示字符串 s 的第 i 到 j 
		个字母组成的串（下文表示成 s[i:j]）是否为回文串：
			P(i,j)= true，子串 Si…Sj 是回文串
					false， 其它情况​
		这里的「其它情况」包含两种可能性：
			（1）s[i,j] 本身不是一个回文串；
			（2）i>j，此时 s[i,j] 本身不合法。
		那么我们就可以写出动态规划的状态转移方程：
			P(i,j)=P(i+1,j−1)∧(Si==Sj)，^表示并
		也就是说，只有 s[i+1:j−1] 是回文串，并且 s 的第 i 和 j 个字母相同时，s[i:j] 才会是回文串。
		
		上文的所有讨论是建立在子串长度大于 2 的前提之上的，我们还需要考虑动态规划中的边界条件，
		即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，
		只要它的两个字母相同，它就是一个回文串。
		因此我们就可以写出动态规划的边界条件：
			P(i,i)=true
			P(i,i+1)=(Si​==Si+1​)​
		根据这个思路，我们就可以完成动态规划了，最终的答案即为所有 
			P(i,j)=true 中 j−i+1（即子串长度）的最大值。
		注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，
			因此一定要注意动态规划的循环顺序。
	时间复杂度：O(n2)
		其中 n 是字符串的长度。动态规划的状态总数为 O(n2)，对于每个状态，我们需要转移的时间为 O(1)。
	空间复杂度：O(n2)
		即存储动态规划状态需要的空间。
*/
func longestPalindromeByDP(s string) string {
	n := len(s)
	ans := ""
	// 记录 s[i:j] 子串是不是回文的状态
	dp := make([][]bool, n)
	for i := 0; i < n; i ++ {
		dp[i] = make([]bool, n)
	}
	// sl 表示当前子串的长度，从 0 开始计数方便 i、j 下标的变化
	for sl := 0; sl < n; sl ++ {
		// i 表示子串起始下标，j 表示子串终止下标
		for i := 0; i + sl < n; i ++ {
			j := i + sl
			if sl == 0 {
				// 只有一个字符
				dp[i][j] = true
			} else if sl == 1 {
				// 有两个字符
				if s[i] == s[j] {
					dp[i][j] = true
				}
			} else {
				// 三个及以上的字符
				if s[i] == s[j] && dp[i + 1][j - 1] {
					dp[i][j] = true
				}
			}
			// 如果当前子串是回文子串，且长度大于已记录的最长回文子串，则更新
			// 此处 sl + 1 是因为 sl 是从 0 开始计数的
			if dp[i][j] && sl + 1 > len(ans) {
				// p(i,j) 表示Si~Sj 包含 j 的，故 s[i:j+1] 需要加1
				ans = s[i:j+1]
			}
		}
	}
	return ans
}

/* 
	方法二：中心扩展算法
	思路与算法：
		我们仔细观察一下方法一中的状态转移方程：
			P(i,i)=true
			P(i,i+1)=(Si==Si+1)
			P(i,j)=P(i+1,j−1)∧(Si==Sj)
		找出其中的状态转移链：
			P(i,j)←P(i+1,j−1)←P(i+2,j−2)←⋯←某一边界情况
		可以发现，所有的状态在转移的时候的可能性都是唯一的。也就是说，
		我们可以从每一种边界情况开始「扩展」，也可以得出所有的状态对应的答案。
		边界情况即为子串长度为 1 或 2 的情况。我们枚举每一种边界情况，并从对应的子串开始不断地向两边扩展。
		如果两边的字母相同，我们就可以继续扩展，例如从 P(i+1,j−1) 扩展到 P(i,j)；
		如果两边的字母不同，我们就可以停止扩展，因为在这之后的子串都不能是回文串了。
		「边界情况」对应的子串实际上就是我们「扩展」出的回文串的「回文中心」。

		方法二的本质即为：我们枚举所有的「回文中心」并尝试「扩展」，直到无法扩展为止，
		此时的回文串长度即为此「回文中心」下的最长回文串长度。
		我们对所有的长度求出最大值，即可得到最终的答案。
	时间复杂度：O(n2)
		其中 n 是字符串的长度。长度为 1 和 2 的回文中心分别有 n 和 n−1 个，每个回文中心最多会向外扩展 O(n) 次。
	空间复杂度：O(1)
*/
func longestPalindromeByCenter(s string) string {
	if s == "" {
		return ""
	}
	start, end := 0,0
	for i := 0; i < len(s); i ++ {
		// 一个字符
		left1, right1 := expandAroundCenter(s, i, i)
		// 两个字符
		left2, right2 := expandAroundCenter(s, i, i + 1)

		if right1 - left1 > end - start {
			start, end = left1, right1
		}
		if right2 - left2 > end - start {
			start, end = left2, right2
		}
	}
	return s[start : end + 1]
}
// 中心扩散
func expandAroundCenter(s string, left, right int) (int, int){
	for left >= 0 && right < len(s) && s[left] == s[right] {
		left --
		right ++
	}
	// 退出循环的时候 left 会小1，right 会大1
	return left + 1, right - 1
}

/* 
====================== 9、翻转字符串里的单词 =========================
给定一个字符串，逐个翻转字符串中的每个单词。

说明：
    无空格字符构成一个 单词 。
    输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
    如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

示例 1：
输入："the sky is blue"
输出："blue is sky the"

示例 2：
输入："  hello world!  "
输出："world! hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
示例 3：

输入："a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

示例 4：
输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"

示例 5：
输入：s = "Alice does not even like bob"
输出："bob like even not does Alice"

提示：
    1 <= s.length <= 104
    s 包含英文大小写字母、数字和空格 ' '
    s 中 至少存在一个 单词

进阶：
    请尝试使用 O(1) 额外空间复杂度的原地解法。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/crmp5/
*/
/* 
	方法一：手写字符串处理API
*/
func reverseWords(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	L, R := 0, n - 1
	// 过滤前面的空格
	for L < n && s[L] == ' ' {
		L ++
	}
	// 过滤后面的空格
	for R >= 0 && s[R] == ' ' {
		R --
	}
	if R - L <= 0 {
		return ""
	}
	// 过滤中间多余的空格
	arr := make([]byte, 0)
	for L <= R {
		if s[L] != ' ' || arr[len(arr) - 1] != ' ' {
			arr = append(arr, s[L])
		}
		L ++
	}
	ans := make([]byte, 0)
	wordEnd := len(arr)
	for i := len(arr) - 1; i >= 0; i -- {
		if arr[i] == ' ' {
			ans = append(ans, arr[i + 1 : wordEnd]...)
			ans = append(ans, ' ')
			wordEnd = i
		}
	}
	// 添加最后一个单词
	ans = append(ans, arr[0 : wordEnd]...)
	return string(ans)
}

/* 
	方法二：调用内部字符串处理API
*/
func reverseWords2(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	strArr := strings.Fields(s)
	reversStringArr(strArr)
	return strings.Join(strArr, " ")
}
func reversStringArr(strArr []string) {
	n := len(strArr)
	if n == 0 {
		return
	}
	for i := 0; i < n >> 1; i ++ {
		strArr[i], strArr[n - 1 - i] = strArr[n - 1 - i], strArr[i]
	}
}

/* 
====================== 10、实现 strStr() =========================
实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，
在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
如果不存在，则返回  -1。

示例 1:
输入: haystack = "hello", needle = "ll"
输出: 2

示例 2:
输入: haystack = "aaaaa", needle = "bba"
输出: -1

说明:

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cm5e2/
*/

/* 
	方法一：双指针滑动串口
	思路：
		遍历长字符串，若当前字符与短字符串的一个字符匹配，
		则从该位置开始逐一匹配长字符串与短字符串的字符，中间遇到不匹配的直接跳出。
	时间复杂度：O((N−L)L)，最优时间复杂度为 O(N)。
		其中 N 为 haystack 字符串的长度，L 为 needle 字符串的长度。
		内循环中比较字符串的复杂度为 L，总共需要比较 (N - L) 次。
	空间复杂度：O(1)。
*/
func strStr(haystack string, needle string) int {
	ln, lh := len(needle), len(haystack)
	if ln == 0 {
		return 0
	}
	i, j := 0, 0
	// 不需要对比完，只需要比较到 haystack 剩余长度小于等于 needle 时就行了
	for i = 0; i < lh - ln + 1; i ++ {
		for j = 0; j < ln; j ++ {
			if haystack[i + j] != needle[j] {
				break
			}
		}
		if j == ln {
			return i
		}
	}
	return -1
}


/* 
====================== 11、反转字符串 =========================
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

示例 1：
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]

示例 2：
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cacxi/
*/
/* 
	双指针法：
		对于长度为 N 的待被反转的字符数组，我们可以观察反转前后下标的变化，
		假设反转前字符数组为 s[0] s[1] s[2] ... s[N - 1]，那么反转后字符数组为 s[N - 1] s[N - 2] ... s[0]。
		比较反转前后下标变化很容易得出 s[i] 的字符与 s[N - 1 - i] 的字符发生了交换的规律，因此我们可以得出如下双指针的解法：
			将 left 指向字符数组首元素，right 指向字符数组尾元素。
			当 left < right：
				交换 s[left] 和 s[right]；
				left 指针右移一位，即 left = left + 1；
				right 指针左移一位，即 right = right - 1。
			当 left >= right，反转结束，返回字符数组即可。
	时间复杂度：O(N)
		其中 N 为字符数组的长度。一共执行了 N/2 次的交换。
	空间复杂度：O(1)
		只使用了常数空间来存放若干变量。
*/
func reverseString(s []byte)  {
	ln := len(s)
	if ln < 2 {
		return
	}
	for left, right := 0, ln - 1; left < right; left, right = left + 1, right - 1  {
		s[left], s[right] = s[right], s[left]
	}
}

/* 
====================== 12、数组拆分 I =========================
给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 
例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。
返回该 最大总和 。

示例 1：
输入：nums = [1,4,3,2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
所以最大总和为 4

示例 2：
输入：nums = [6,2,6,5,1,2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c24he/
*/
/* 
	方法一：排序 
	算法：
		为了理解这种方法，让我们从不同的角度来看待问题。我们需要形成数组元​​素的配对，
		使得这种配对中最小的总和最大。因此，我们可以查看选择配对中最小值的操作，
		比如 (a,b) 可能会产生的最大损失 a−b (如果 a>b)。
		如果这类配对产生的总损失最小化，那么总金额现在将达到最大值。
		只有当为配对选择的数字比数组的其他元素更接近彼此时，才有可能将每个配对中的损失最小化。
		考虑到这一点，我们可以对给定数组的元素进行排序，并直接按排序顺序形成元素的配对。
		这将导致元素的配对，它们之间的差异最小，从而导致所需总和的最大化。
	时间复杂度：O(nlog(n))
		排序需要 O(nlog(n)) 的时间。另外会有一次数组的遍历。
	空间复杂度：O(1)。
		仅仅需要常数级的空间.

*/
func arrayPairSum(nums []int) int {
	ln := len(nums)
	if ln == 0 {
		return 0
	}
	sort.Ints(nums)
	sum := 0
	for i := 0; i < ln; i += 2 {
		// 排序后 0, 2, 4, ... 比 1, 3, 5, .. 小
		sum += nums[i]
	}
	return sum
}

/* 
====================== 13、两数之和 II - 输入有序数组 =========================
给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。

说明:
    返回的下标值（index1 和 index2）不是从零开始的。
    你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例:
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cnkjg/
*/
/* 
	方法一：二分查找
	思路：
		先固定其中一个数，另一个数用二分查找
	时间复杂度：O(nlog⁡n)
		其中 n 是数组的长度。需要遍历数组一次确定第一个数，时间复杂度是 O(n)，
		寻找第二个数使用二分查找，时间复杂度是 O(log⁡n)，因此总时间复杂度是 O(nlog⁡n)
	空间复杂度：O(1)
*/
func twoSum(numbers []int, target int) []int {
	ln := len(numbers)
	if ln == 0 {
		return []int{}
	}
	// 先固定一个数，另一个数用二分查找
	for i := 0; i < ln; i ++ {
		t := target - numbers[i]
		j := binarySearch(numbers, i + 1, ln - 1, t)
		if j != - 1 {
			return []int{i + 1, j + 1}
		}
	}
	return []int{}
}
func binarySearch(numbers []int, L, R, target int) int {
	for L + 1 < R {
		mid := (L + R) >> 1
		if numbers[mid] < target {
			L = mid
		} else {
			R = mid
		}
	}
    if numbers[L] == target {
        return L
    }
    if numbers[R] == target {
        return R
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
func twoSum2(numbers []int, target int) []int {
	ln := len(numbers)
	if ln == 0 {
		return []int{}
	}
	L, R := 0, ln - 1
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
	return []int{}
}

/* 
====================== 14、移除元素 =========================
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
示例 1:
给定 nums = [3,2,2,3], val = 3,
函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
你不需要考虑数组中超出新长度后面的元素。

示例 2:
给定 nums = [0,1,2,2,3,0,4,2], val = 2,
函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
注意这五个元素可为任意顺序。
你不需要考虑数组中超出新长度后面的元素。

说明:
为什么返回数值是整数，但输出的答案是数组呢?
请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
你可以想象内部操作如下:
	// nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
	int len = removeElement(nums, val);
	// 在函数里修改输入数组对于调用者是可见的。
	// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
	for (int i = 0; i < len; i++) {
		print(nums[i]);
	}

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cwuyj/
*/
/* 
	方法一：快慢指针法
	思路：
		既然问题要求我们就地删除给定值的所有元素，我们就必须用 O(1) 的额外空间来处理它。
		如何解决？我们可以保留两个指针 i 和 j，其中 i 是慢指针，j 是快指针。
		当 nums[j] 与给定的值相等时，递增 j 以跳过该元素。只要 nums[j]≠val，
		我们就复制 nums[j] 到 nums[i] 并同时递增两个索引。重复这一过程，
		直到 j 到达数组的末尾，该数组的新长度为 i。
	缺点：当数字很长，且要删除的元素个数很少时，差不多相当于对每一个元素都需要进行移动，
		如此效率很低下。
	时间复杂度：O(n)
		假设数组总共有 n 个元素，i 和 j 至少遍历 2n 步。
	空间复杂度：O(1)
*/
func removeElement(nums []int, val int) int {
	slow, fast := 0, 0
	for fast < len(nums) {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow ++
		}
		fast ++
	}
	return slow
}

/* 
	方法二：双指针 —— 当要删除的元素很少时
	思路：
		现在考虑数组包含很少的要删除的元素的情况。例如，num=[1，2，3，5，4]，Val=4。
		之前的算法会对前四个元素做不必要的复制操作。另一个例子是 num=[4，1，2，3，5]，Val=4。
		似乎没有必要将 [1，2，3，5] 这几个元素左移一步，因为问题描述中提到元素的顺序可以更改。
	算法：
		当我们遇到 nums[i]=val 时，我们可以将当前元素与最后一个元素进行交换，
		并释放最后一个元素。这实际上使数组的大小减少了 1。
		请注意，被交换的最后一个元素可能是您想要移除的值。
		但是不要担心，在下一次迭代中，我们仍然会检查这个元素。
	时间复杂度：O(n))
		i 和 n 最多遍历 n 步。在这个方法中，赋值操作的次数等于要删除的元素的数量。
		因此，如果要移除的元素很少，效率会更高。
	空间复杂度：O(1)
*/
func removeElement2(nums []int, val int) int {
	i, n := 0, len(nums)
	for i < n {
		if nums[i] == val {
			// 把最后一个元素放到当前位置
			nums[i] = nums[n - 1]
			// 释放最后一个元素
			n --
		} else {
			// 因为最后一个元素也有可能时要被删除的，所以不能跳过，
			// 必须用 else 从 i 开始
			i ++
		}
	}
	return i
}

/* 
====================== 15、最大连续1的个数 =========================
给定一个二进制数组， 计算其中最大连续1的个数。

示例 1:
输入: [1,1,0,1,1,1]
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.

注意：
    输入的数组只包含 0 和1。
    输入数组的长度是正整数，且不超过 10,000。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cd71t/
*/
/* 
	方法一：一次遍历
	算法：
		用一个计数器 count 记录 1 的数量，另一个计数器 maxCount 记录当前最大的 1 的数量。
		当我们遇到 1 时，count 加一。
		当我们遇到 0 时：
			将 count 与 maxCount 比较，maxCoiunt 记录较大值。
			将 count 设为 0。
		返回 maxCount。
	时间复杂度：O(N)
		N 值得是数组的长度。
	空间复杂度：O(1)
		仅仅使用了 count 和 maxCount。
*/
func findMaxConsecutiveOnes(nums []int) int {
	ln := len(nums)
	if ln == 0 {
		return 0
	}
	cnt, maxCount := 0, 0
	for i := 0; i < ln; i ++ {
		if nums[i] == 1 {
			cnt ++
		} else {
			maxCount = max(maxCount, cnt)
			cnt = 0
		}
	}
	return max(maxCount, cnt)
}

/* 
	方法二：双指针
	思路：
		让 left、right指针同时指向第一个1，之后遇到 1 时 right ++，
		遇到 0 时计算 right - left，即为当前连续1的个数，再与最大值比较和更新最大值。
		然后从 right 之后重置 left、right指针到新的 1，重复上述过程找出最大的连续1。
	时间复杂度：O(N)
		N 值得是数组的长度。
	空间复杂度：O(1)
*/
func findMaxConsecutiveOnes2(nums []int) int {
	left := findLeft(nums, 0)
	right := left
	ans := 0
	ln := len(nums)
	for right < ln {
		if nums[right] != 1 {
			ans = max(ans, right - left)
			// 从 right 的下一个开始找
			left = findLeft(nums, right + 1)
			right = left
		} else {
			right ++
		}
	}
	// 当最后面是连续的 1 时需要计算
	if nums[right - 1] == 1 {
		ans = max(ans, right - left)
	}
	return ans
}
func findLeft(nums []int, left int) int {
	for left < len(nums) {
		if nums[left] == 1 {
			break
		}
		left ++
	}
	return left
}
// func max(a, b int) int {
// 	if a > b {
// 		return a
// 	}
// 	return b
// }

/* 
====================== 16、长度最小的子数组 =========================
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

示例：
输入：s = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。

进阶：
    如果你已经完成了 O(n) 时间复杂度的解法, 请尝试 O(n log n) 时间复杂度的解法。
作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c0w4r/
*/

/* 
	方法一：暴力解法
	思路：
		初始化子数组的最小长度为无穷大，枚举数组 nums 中的每个下标作为子数组的开始下标，
		对于每个开始下标 i，需要找到大于或等于 i 的最小下标 j，使得从 nums[i] 到 nums[j]
		的元素和大于或等于 s，并更新子数组的最小长度（此时子数组的长度是 j−i+1）。
	时间复杂度：O(n^2)
		其中 n 是数组的长度。需要遍历每个下标作为子数组的开始下标，
		对于每个开始下标，需要遍历其后面的下标得到长度最小的子数组。
	空间复杂度：O(1)
*/
func minSubArrayLen(s int, nums []int) int {
	ln := len(nums)
	if ln == 0 {
		return 0
	}
	minLen := 1 << 63 - 1
	for i := 0; i < ln; i ++ {
		sum := 0
		for j := i; j < ln; j ++ {
			sum += nums[j]
			if sum >= s {
				minLen = min(minLen, j - i + 1)
				break
			}
		}
	}
	// 找不到
	if minLen == 1 << 63-1 {
		return 0
	}
	return minLen
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 
	方法二：双指针
	思路：
		在方法一中，都是每次确定子数组的开始下标，然后得到长度最小的子数组，
		因此时间复杂度较高。为了降低时间复杂度，可以使用双指针的方法。
		定义两个指针 start 和 end 分别表示子数组的开始位置和结束位置，
		维护变量 sum 存储子数组中的元素和（即从 nums[start] 到 nums[end] 的元素和）。
		初始状态下，start 和 end 都指向下标 0，sum 的值为 0。
		每一轮迭代，将 nums[end] 加到 sum，如果 sum≥s，则更新子数组的最小长度
		（此时子数组的长度是 end−start+1），然后将 nums[start] 从 sum 中减去
		并将 start 右移，直到 sum<s，在此过程中同样更新子数组的最小长度。
		在每一轮迭代的最后，将 end 右移。
	时间复杂度：O(n)
		其中 n 是数组的长度。指针 start 和 end 最多各移动 n 次。
	空间复杂度：O(1)
*/
func minSubArrayLen2(s int, nums []int) int {
	start, end := 0, 0
	Max := 1 << 63 - 1
	minLen, sum := Max, 0
	for end < len(nums) {
		sum += nums[end]
		for sum >= s {
			minLen = min(minLen, end - start + 1)
			sum -= nums[start]
			start ++
		}
		end ++
	}
	if minLen == Max {
		return 0
	}
	return minLen
}

/* 
====================== 17、杨辉三角 =========================
	  1
     1 1
    1 2 1
   1 3 3 1
  1 4 6 4 1
 1 5 10 10 5 1

在杨辉三角中，每个数是它左上方和右上方的数的和。
给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。

示例:

输入: 5
输出:
[
[1],
[1,1],
[1,2,1],
[1,3,3,1],
[1,4,6,4,1]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cuj3m/
*/
/* 
	算法：动态规划（DP）
	思路：
		在杨辉三角中只要知道一行数据，就可以推出下一行的数据
		推导函数： 
			/ nums[r][c] = 1, c == 0 或 c == r
			\ nums[r][c] = nums[r-1][c-1] + nums[r-1][c]
	时间复杂度：O(numRows^2)
		虽然更新 triangle 中的每个值都是在常量时间内发生的，
		但它会被执行 O(numRows^2) 次。想要了解原因，就需要考虑总共有多少次循环迭代。
		很明显外层循环需要运行 numRows 次，但在外层循环的每次迭代中，
		内层循环要运行 rowNum 次。因此，triangle 发生的更新总数为
		1+2+3+…+numRows，根据高斯公式有：
			1+2+3+...+numRows = 1/2*[(numRows + 1) * numRows]
							  = 1/2*(numRows^2) + 1/2*(numRows)
							  = O(numRows^2)
	空间复杂度：O(numRows^2)
*/
func generate(numRows int) [][]int {
	if numRows == 0 {
		return [][]int{}
	}
	res := make([][]int, numRows)
	res[0] = []int{1}
	if numRows == 1 {
		return res
	}
	for i := 1; i < numRows; i ++ {
		list := make([]int, i + 1)
		list[0] = 1		// 第一列
		for j := 1; j < i; j ++ {
			list[j] = res[i - 1][j - 1] + res[i - 1][j]
		}
		list[i] = 1		// 最后一列
		res[i] = list
	}
	return res
}

/* 
====================== 18、杨辉三角 II =========================
给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行，k 从0开始。
示例:

输入: 3
输出: [1,3,3,1]

进阶：
你可以优化你的算法到 O(k) 空间复杂度吗？
*/
/* 
	方法一：动态规划
	思路：
		使用一个额外数组记录上一次推导的结果
	时间复杂度：O(k^2)
	空间复杂度：O(2K - 1)
		我们需要一个 k-1 的数组记录上一次推导的结果，并返回 k 个元素的结果
*/
func getRow(rowIndex int) []int {
	res := make([]int, rowIndex + 1)
	res[0] = 1
	if rowIndex == 0 {
		return res
	}
	// 使用 copy 时需要先初始化长度
	preList := make([]int, rowIndex + 1)
	copy(preList, res)
	for i := 1; i <= rowIndex; i ++ {
		res[0] = 1
		for j := 1; j < i; j ++ {
			res[j] = preList[j - 1] + preList[j]
		}
		res[i] = 1
		copy(preList, res)
	}
	return res
}

/* 
	方法二：动态规划
	思路：
		只用一个数组保存结果，但需要从中间向两边计算，
		因为 ans[j] = ans[j] + ans[j-1] 用到了 j-1 在计算 j 的时候， 
		j-1 位置不能先改变。
	时间复杂度：O(k^2)
	空间复杂度：O(k)
		只用一个 K 长度的数组保存结果并返回
*/
func getRow2(rowIndex int) []int {
	res := make([]int, rowIndex + 1)
	res[0] = 1
	if rowIndex == 0 {
		return res
	}
	for i := 1; i <= rowIndex; i ++ {
		res[0] = 1
		// 需要从中间向两边计算， 
		// 因为 ans[j] = ans[j] + ans[j-1] 用到了 j-1 在计算 j 的时候， 
		// j-1位置不能先改变
		for j := i >> 1; j >= 1; j -- {
			res[j] = res[j - 1] + res[j]
			// 从中间隔开，与 j 对称的位置其值与 j 相同
			res[i - j] = res[j]
		}
		res[i] = 1
	}
	return res
}

/* 
====================== 19、反转字符串中的单词 III =========================
给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例：
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"

提示：
    在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c8su7/
*/
/* 
	方法一：分割字符串成数组
	思路：
		字符串分割成单词，然后逐个反转。
	时间复杂度：O(kn)
		k 是单词个数，n 是字符串长度，我们需要先将字符串转为单词数组，再遍历数组进行单个单词的反转。
	空间复杂度：O(kn)
		k 是单词个数，n 是字符串长度，我们需要把字符串转为单词数组，再把每一个单词转为字符数组。
*/
func reverseWordsIII(s string) string {
	if len(s) == 0 {
		return ""
	}
	arr := strings.Fields(s)
	for i := 0; i < len(arr); i ++ {
		arr[i] = reverseWord(arr[i])
	}
	return strings.Join(arr, " ")
}
func reverseWord(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	arr := []byte(s)
	for i := 0; i < n >> 1; i ++ {
		arr[i], arr[n - 1 - i] = arr[n - 1 - i], arr[i]
	}
	return string(arr)
}

/* 
	方法二：双指针
	思路：
		使用双指针一个指向单词的第一个字符，一个指向最后一个字符，然后翻转。
	时间复杂度：O(n)
		n 是字符串的长度，我们需要对字符串进行一次完整遍历
	空间复杂度：O(n)
		n 是字符串的长度，我们需要把字符串换成数组再进行交换操作。
*/
func reverseWordsIII2(s string) string {
	if len(s) == 0 {
		return s
	}
	arr := []byte(s)
	// 在最后添加一个哨兵，循环退出后就不用再单独处理最后一个单词了
	arr = append(arr, ' ')
	L,R := 0, 0
	ln := len(arr)
	for R < ln {
		if arr[R] == ' ' {
			// 此时 R 指向 ' '，所以 n 不用 + 1
			n := R - L
			for i := 0; i < n >> 1; i ++ {
				arr[L + i], arr[L + (n - 1 - i)] = arr[L + (n - 1 - i)], arr[L + i]
			}
			// L 指向下一个单词的开头
			L = R + 1
		}
		R ++
	}
	// 去除添加的哨兵
	return string(arr[: ln - 1])
}


/* 
====================== 20、删除排序数组中的重复项 =========================
给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

示例 1:
给定数组 nums = [1,1,2], 
函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 
你不需要考虑数组中超出新长度后面的元素。

示例 2:
给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cq376/
*/
/* 
	算法：双指针
	思路：
		数组完成排序后，我们可以放置两个指针 i 和 j，其中 i 是慢指针，而 j 是快指针。
		只要 nums[i]=nums[j]，我们就增加 j 以跳过重复项。当我们遇到 nums[j]≠nums[i] 时，
		跳过重复项的运行已经结束，因此我们必须把它（nums[j]）的值复制到 nums[i+1]。
		然后递增 i，接着我们将再次重复相同的过程，直到 j 到达数组的末尾为止。
	时间复杂度：O(n)，
		假设数组的长度是 n，那么 i 和 j 分别最多遍历 n 步。
	空间复杂度：O(1)
*/
func removeDuplicates(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	if n == 1 {
		return 1
	}
	L, R := 0, 0
	for R < n {
		if nums[L] != nums[R] {
			L ++
			nums[L], nums[R] = nums[R], nums[L]
		}
		R ++
	}
	// 因为调用是 nums[0:L]，所以返回时 L 需要加1
	return L + 1
}

/* 
====================== 21、移动零 =========================
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
    必须在原数组上操作，不能拷贝额外的数组。
    尽量减少操作次数。
*/
/* 
	算法：双指针
	思路：
		左右指针初始化在数组头部，右指针循环右移，每当右指针指向非零时，
		交换左右指针指向的元素，同时左指针右移。
	注意：
		此方法中同时指向非零时，它们的位置是一致的，即指向相同元素，
		只会和自己交换而不存在非零元素被移到后面的情况。
		因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。
	时间复杂度：O(n)
		其中 n 为序列长度。每个位置至多被遍历两次。
	空间复杂度：O(1)
		只需要常数的空间存放若干变量。
*/
func moveZeroes(nums []int)  {
	n := len(nums)
	if n == 0 || n == 1{
		return
	}
	L, R := 0, 0
	for R < n {
		if nums[R] != 0 {
			nums[L], nums[R] = nums[R], nums[L]
			L ++
		}
		R ++
	}
}






// ====================== 案列测试 =========================

// 1、测试寻找数组的中心索引 
func pivotIndexTest() {
	nums := []int{1, 7, 3, 6, 5, 6}
	res := pivotIndex(nums)
	fmt.Println(res)
}

// 2、测试搜索插入位置
func searchInsertTest() {
	nums := []int{1,3,5,6}
	res := searchInsert(nums, 7)
	fmt.Println(res)
}

// 3、测试合并区间
func mergeTest() {
	nums := [][]int{
		{1,3},
		{2,6},
		{8,10},
		{15,18},
	}
	res := merge(nums)
	fmt.Println(res)
}

// 4、测试旋转矩阵
func rotateTest() {
	matrix := [][]int{
		{1,2,3},
		{4,5,6},
		{7,8,9},
	}
	rotate(matrix)
	fmt.Println(matrix)
}

// 5、测试零矩阵
func setZeroesTest() {
	matrix := [][]int {
		{1,0,3},
	}
	setZeroes(matrix)
	fmt.Println(matrix)
}

// 7、测试最长公共前缀
func longestCommonPrefixTest() {
	strs := []string{"flower","flow","flight"}
	res := longestCommonPrefix(strs)
	fmt.Println(res)
}

// 8、测试最长回文子串
func longestPalindromeTest() {
	str := "babad"
	// res := longestPalindromeByDP(str)
	res := longestPalindromeByCenter(str)
	fmt.Println(res)
}

// 9、测试翻转字符串里的单词
func reverseWordTest() {
	s := "the sky is blue"
	// res := reverseWords(s)
	res := reverseWords2(s)
	fmt.Println(res)
}

// 6、测试实现 strStr()
func strStrTest() {
	s1 := "aaaaa"
	s2 := "bba"
	res := strStr(s1, s2)
	fmt.Println(res)
}

// 7、测试反转字符串
func reverseStringTest() {
	s := "abcde"
	arr := []byte(s)
	reverseString(arr)
	fmt.Printf("%s\n", arr)
}

// 8、测试数组拆分 I
func arrayPairSumTest() {
	nums := []int{6,2,6,5,1,2}
	res := arrayPairSum(nums)
	fmt.Println(res)
}

// 9、测试两数之和 II - 输入有序数组
func twoSumTest() {
	nums := []int{5,25,75}
	res := twoSum(nums, 100)
	// res := twoSum2(nums, 100)
	fmt.Println(res)
}

// 10、测试移除元素
func removeElementTest() {
	nums := []int{3,2,2,3}
	// res := removeElement(nums, 3)
	res := removeElement2(nums, 3)
	fmt.Println(res)
	fmt.Println(nums[:res])
}

// 11、测试最大连续1的个数
func findMaxConsecutiveOnesTest() {
	nums := []int{1,1,0,1,1,1}
	// res := findMaxConsecutiveOnes(nums)
	res := findMaxConsecutiveOnes2(nums)
	fmt.Println(res)
}

// 12、测试长度最小的子数组
func minSubArrayLenTest() {
	nums := []int{2,3,1,2,4,3}
	// res := minSubArrayLen(7, nums)
	res := minSubArrayLen2(7, nums)
	fmt.Println(res)
}

// 13、测试杨辉三角
func generateTest() {
	res := generate(5)
	fmt.Println(res)
}

// 14、测试杨辉三角II
func getRowTest() {
	res := getRow2(3)
	fmt.Println(res)
}

// 15、测试反转字符串中的单词 III
func reverseWordsTest() {
	s := "Let's take LeetCode contest"
	// res := reverseWords(s)
	res := reverseWords2(s)
	fmt.Println(res)
}

// 16、测试删除排序数组中的重复项
func removeDuplicatesTest() {
	nums := []int{0,0,1,1,1,2,2,3,3,4}
	res := removeDuplicates(nums)
	fmt.Println(nums[:res])
}

// 17、测试移动零
func moveZeroesTest() {
	nums := []int{0,1,0,3,12}
	moveZeroes(nums)
	fmt.Println(nums)
}

func main() {
	// pivotIndexTest()
	// searchInsertTest()
	// mergeTest()
	// rotateTest()
	// setZeroesTest()
	// longestCommonPrefixTest()
	// longestPalindromeTest()
	// reverseWordTest()
	// strStrTest()
	// reverseStringTest()
	// arrayPairSumTest()
	twoSumTest()
	// removeElementTest()
	// findMaxConsecutiveOnesTest()
	// minSubArrayLenTest()
	// generateTest()
	// getRowTest()
	// reverseWordsTest()
	// removeDuplicatesTest()
	// moveZeroesTest()
}