# 树

## 递归

一棵树要么是空树，要么有两个指针，每个指针指向一棵树。树是一种递归结构，很多树的问题可以使用递归来处理。

### 14. 找出二叉树中第二小的节点

[671. Second Minimum Node In a Binary Tree (Easy)](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/)

```
Input:
   2
  / \
 2   5
    / \
    5  7

Output: 5
```

一个节点要么具有 0 个或 2 个子节点，如果有子节点，那么根节点是最小的节点。

```java
//根据题意，这题可以转换成求，左右子节点中的最小值．

class Solution {
    public int findSecondMinimumValue(TreeNode root) {
     return traversal(root,root.val);
    }
    
    private int traversal(TreeNode root,int value){
        if(root == null){
            return -1;
        }
        if(root.val > value){
            return root.val;
        }
        // 寻找左右子节点中，第一个大于自己的节点
        int l = traversal(root.left,value);
        int r = traversal(root.right,value);

        // 存在两个子节点
        if(l>=0 && r>=0){
            return Math.min(l,r);
        }
        //　存在0个子节点
        return Math.max(l,r);
    }
  
}
```

# 层次遍历

使用 BFS 进行层次遍历。不需要使用两个队列来分别存储当前层的节点和下一层的节点，因为在开始遍历一层的节点时，当前队列中的节点数就是当前层的节点数，只要控制遍历这么多节点数，就能保证这次遍历的都是当前层的节点。

## 1. 一棵树每层节点的平均数

[637. Average of Levels in Binary Tree (Easy)](https://leetcode.com/problems/average-of-levels-in-binary-tree/description/)

```
public List<Double> averageOfLevels(TreeNode root) {
    List<Double> ret = new ArrayList<>();
    if (root == null) return ret;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
        int cnt = queue.size();
        double sum = 0;
        for (int i = 0; i < cnt; i++) {
            TreeNode node = queue.poll();
            sum += node.val;
            if (node.left != null) queue.add(node.left);
            if (node.right != null) queue.add(node.right);
        }
        ret.add(sum / cnt);
    }
    return ret;
}
```