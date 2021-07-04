package main

/*
============== 剑指 Offer 05. 替换空格 ==============
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

示例 1：
输入：s = "We are happy."
输出："We%20are%20happy."

限制：
0 <= s 的长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof
*/
/*
	方法一：复制
	思路：
		把字符串看成字符数组，使用另一个数组对其进行逐一复制，遇到空格则复制
		为对应的替换字符。
	时间复杂度：O(n)
		n 是字符串的长度。
	空间复杂度：O(n)
		n 是字符串的长度，我们需要完整复制原字符串，并把空格替换成 %20。
*/
func replaceSpace(s string) string {
	if len(s) == 0 {
		return s
	}
	ans := []rune{}
	for _, v := range s {
		if v == ' ' {
			ans = append(ans, []rune("%20")...)
			continue
		}
		ans = append(ans, v)
	}
	return string(ans)
}

/*
	方法二：不使用 slice
	思路：
		不使用切片，那就使用数组，提前开辟好足够的数组空间即可。
		我们可以假设所有字符都为空格，那么我们所需的空间即为 n*3，但 golang
		不支持用变量来创建数组，所以我们只能从字符串的最大长度 10000 入手，
		创建一个 10000*3 的数组来存储结果，然后对原数组进行逐一复制和替换。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func replaceSpace2(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	// 0 <= s 的长度 <= 10000
	ans := [3 * 10000]rune{}
	i := 0
	for _, v := range s {
		if v == ' ' {
			ans[i] = '%'
			i++
			ans[i] = '2'
			i++
			ans[i] = '0'
			i++
			continue
		}
		ans[i] = v
		i++
	}
	return string(ans[:i])
}

/*
	方法三：双指针
	思路：
		我们先遍历一次字符串，计算出空格的数量 emptyCnt，然后创建一个长度为
		n + 2*emptyCnt 的数组，因为每一个空格都要被替换为三个字符 %20
		接着我们用两个指针，一个指向我们创建的数组的尾部，另一个指向字符串 s
		的尾部，从尾到头一一复制 s 的字符，遇到空格时替换为 %20
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func replaceSpace3(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	emptyCnt := 0
	for _, v := range s {
		if v == ' ' {
			emptyCnt++
		}
	}
	ans := make([]byte, n+2*emptyCnt)
	i, j := n-1, n+2*emptyCnt-1
	for i >= 0 {
		if s[i] == ' ' {
			ans[j] = '0'
			j--
			ans[j] = '2'
			j--
			ans[j] = '%'
			j--
			i--
			continue
		}
		ans[j] = s[i]
		j--
		i--
	}
	return string(ans)
}

/* 
	相关题目：
		有两个排序的数组 A1 和 A2，其中 A1 的末尾有足够多的空余空间容纳 A2，
		请事先一个函数，把 A2 中的所有数字插入 A1 中，并且所有的数字都是排
		序的。
	思路：
		使用双指针从尾到头比较 A1、A2 的数字，把较大的数字先复制到 A1 尾部
		合适的位置。
*/