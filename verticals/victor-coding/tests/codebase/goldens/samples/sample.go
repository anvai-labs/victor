package main

import "fmt"

func publicFn(items []string) []string {
	results := make([]string, 0)
	for _, item := range items {
		if item == "" {
			continue
		}
		results = append(results, item)
	}
	return results
}

func main() {
	fmt.Println(publicFn([]string{"a", "", "c"}))
}
