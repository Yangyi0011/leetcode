package main

import (
	"fmt"
)

/* 
========================== 滑动窗口算法思想 =============================
	在滑动窗口类型的问题中都会有两个指针。
	一个用于「延伸」现有窗口的 R 指针，和一个用于「收缩」窗口的 L 指针。在任意时刻，
	只有一个指针运动，而另一个保持静止。我们在 s 上滑动窗口，通过移动 R 指针不断扩
	张窗口。当窗口包含 t 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。

	滑动窗口算法模板：
	void slidingWindow(string s, string t) {
		unordered_map<char, int> need, window;
		for (char c : t) need[c]++;

		int left = 0, right = 0;
		int valid = 0;
		while (right < s.size()) {
			// c 是将移入窗口的字符
			char c = s[right];
			// 右移窗口
			right++;
			// 进行窗口内数据的一系列更新
			...

			// debug 输出的位置
			printf("window: [%d, %d)\n", left, right);

			// 判断左侧窗口是否要收缩
			while (window needs shrink) {
				// d 是将移出窗口的字符
				char d = s[left];
				// 左移窗口
				left++;
				// 进行窗口内数据的一系列更新
				...
			}
		}
	}
*/

/* 
=================== 1、最小覆盖子串 ===================
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。
如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"

示例 2：
输入：s = "a", t = "a"
输出："a"

提示：

    1 <= s.length, t.length <= 105
    s 和 t 由英文字母组成
 
进阶：你能设计一个在 o(n) 时间内解决此问题的算法吗？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/minimum-window-substring
*/
/* 
	方法一：滑动窗口思想
	思路：
		本问题要求我们返回字符串 s 中包含字符串 t 的全部字符的最小窗口。
		我们称包含 t 的全部字母的窗口为「可行」窗口。
		我们可以用滑动窗口的思想解决这个问题，在滑动窗口类型的问题中都会有两个指针。
		一个用于「延伸」现有窗口的 R 指针，和一个用于「收缩」窗口的 L 指针。在任意时刻，
		只有一个指针运动，而另一个保持静止。我们在 s 上滑动窗口，通过移动 R 指针不断扩
		张窗口。当窗口包含 t 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。
		
		如何判断当前的窗口包含所有 t 所需的字符呢？我们可以用一个哈希表表示 t 中所有
		的字符以及它们的个数，用一个哈希表动态维护窗口中所有的字符以及它们的个数，
		如果这个动态表中包含 t 的哈希表中的所有字符，并且对应的个数都不小于 t 的哈
		希表中各个字符的个数，那么当前的窗口是「可行」的。
		注意：这里 t 中可能出现重复的字符，所以我们要记录字符的个数。
	时间复杂度：O(C⋅∣s∣+∣t∣)
		最坏情况下左右指针对 s 的每个元素各遍历一遍，哈希表中对 s 中的每个元素各插入、
		删除一次，对 t 中的元素各插入一次。每次检查是否可行会遍历整个 t 的哈希表，
		哈希表的大小与字符集的大小有关，设字符集大小为 C，则渐进时间复杂度为 O(C⋅∣s∣+∣t∣)。
	空间复杂度：O(C)
		这里用了两张哈希表作为辅助空间，每张哈希表最多不会存放超过字符集大小的键值对，
		我们设字符集大小为 C ，则渐进空间复杂度为 O(C)。
*/
func minWindow(s string, t string) string {
	sn, tn := len(s), len(t)
	if tn > sn {
		return ""
	}
	// 初始化需要的字符及其出现次数
	need := make(map[byte]int)
	for i := 0; i < tn; i ++ {
		need[t[i]] ++
	}
	// 初始化窗口及其左右指针
	window := make(map[byte]int)
	L, R := 0, 0
	// 初始化结果字符的起始和结束位置
	start, end := 0, 0
	// 初始化符合条件的最小窗口长度
	min := 1 << 63 - 1
	// 初始化需要匹配的字符数
	match := 0
	for R < sn {
		// 窗口向右扩张
		c := s[R]
		R ++
		// 如果是需要的字符，则把它放入窗口中
		if need[c] > 0 {
			window[c] ++
			// 如果当前字符的出现次数与需要的次数相匹配，则完成一个字符的匹配
			if window[c] == need[c] {
				match ++
			}
		}
		//  字符都匹配完成后，可以尝试着收缩窗口长度
		for match == len(need) {
			// 获取最小窗口
			if R - L < min {
				min = R - L
				start, end = L, R
			}
			// 收缩窗口
			c = s[L]
			L ++
			// 如果左边丢掉的字符是需要的字符，则其在窗口中出现的次数 -1
			if need[c] > 0 {
				// 如果丢弃字符前该字符的出现次数刚好等于需要次数，
				// 则丢弃后就无法满足匹配了，故完成匹配的字符数 -1
				if need[c] == window[c] {
					match --
				}
				window[c] --
			}
		}
	}
	if min == (1 << 63) - 1 {
		return ""
	}
	return s[start : end]
}

// ========================== 案列测试 =============================
func minWindowTest() {
	s := "a"
	t := "a"
	res := minWindow(s, t)
	fmt.Println(res)
}

func main() {
	minWindowTest()
}