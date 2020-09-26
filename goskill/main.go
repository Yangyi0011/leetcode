package main

import (
	"fmt"
)

/* 
	golang 使用技巧
*/
func main() {
	// stackFunc()
	queueFunc()
}

// 栈
// go 可以用切片来模拟栈操作
func stackFunc() {

	// 创建栈
	stack := make([]int, 0)
	
	// 压栈
	push := func(val int) {
		stack = append(stack, val)
	}

	// 栈是否为空
	isEmpty := func() bool {
		return len(stack) == 0
	}

	// 出栈
	pop := func() int {
		if isEmpty() {
			fmt.Println("栈为空！")
			return 0
		}

		res := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]
		return res
	}

	// 测试
	for {
		fmt.Printf("1、压栈\n2、出栈\n0、退出\n请选择：")
		var flag int
		fmt.Scanln(&flag)
		switch flag {
			case 0:{
				return
			}
			case 1: {
				fmt.Printf("请输入要压栈的数：")
				var val int
				fmt.Scanln(&val)
				push(val)
				fmt.Println("当前栈：", stack)
			}
			case 2: {
				val := pop()
				fmt.Println("出栈元素：", val)
				fmt.Println("当前栈：", stack)
			}
		}
		// 每轮操作完成空一行，美观
		fmt.Println()
	}
}

// 队列
// go 可以用切片来模拟队列操作
func queueFunc() {
	// 创建一个队列
	queue := make([]int, 0)

	// 入队
	push := func(val int) {
		queue = append(queue, val)
	}

	// 队列是否为空
	isEmpty := func() bool {
		return len(queue) == 0
	}

	// 出队
	pop := func() int {
		if isEmpty() {
			fmt.Println("队列为空！")
			return 0
		}

		val := queue[0]
		queue = queue[1:]
		return val
	}

	// 测试
	for {
		fmt.Printf("1、入队\n2、出队\n0、退出\n请选择：")
		var flag int
		fmt.Scanln(&flag)
		switch flag {
			case 0:{
				return
			}
			case 1: {
				fmt.Printf("请输入要入队的数：")
				var val int
				fmt.Scanln(&val)
				push(val)
				fmt.Println("当前队列：", queue)
			}
			case 2: {
				val := pop()
				fmt.Println("出队元素：", val)
				fmt.Println("当前队列：", queue)
			}
		}
		// 每轮操作完成空一行，美观
		fmt.Println()
	}
}