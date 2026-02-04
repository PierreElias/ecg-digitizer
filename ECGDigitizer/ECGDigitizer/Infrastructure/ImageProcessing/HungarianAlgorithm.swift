import Foundation
import Accelerate

/// Hungarian algorithm (Munkres algorithm) for optimal assignment
///
/// Given an n×m cost matrix, finds the optimal assignment that minimizes
/// total cost. Used for matching ECG line segments to leads.
///
/// Time complexity: O(n³) where n = max(rows, cols)
/// Space complexity: O(n²)
struct HungarianAlgorithm {

    /// Find optimal assignment for a cost matrix
    /// - Parameter costMatrix: n×m matrix where costMatrix[i][j] is cost of assigning row i to column j
    /// - Returns: Array of (row, column) assignments
    static func solve(costMatrix: [[Float]]) -> [(row: Int, col: Int)] {
        guard !costMatrix.isEmpty && !costMatrix[0].isEmpty else { return [] }

        let numRows = costMatrix.count
        let numCols = costMatrix[0].count
        let n = max(numRows, numCols)

        // Pad matrix to square if needed
        var matrix = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        for i in 0..<numRows {
            for j in 0..<numCols {
                matrix[i][j] = costMatrix[i][j]
            }
        }

        // Step 1: Subtract row minimums
        for i in 0..<n {
            let minVal = matrix[i].min() ?? 0
            for j in 0..<n {
                matrix[i][j] -= minVal
            }
        }

        // Step 2: Subtract column minimums
        for j in 0..<n {
            var minVal: Float = .infinity
            for i in 0..<n {
                minVal = min(minVal, matrix[i][j])
            }
            for i in 0..<n {
                matrix[i][j] -= minVal
            }
        }

        // Initialize tracking arrays
        var rowCovered = [Bool](repeating: false, count: n)
        var colCovered = [Bool](repeating: false, count: n)
        var starred = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
        var primed = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)

        // Step 3: Star zeros
        for i in 0..<n {
            for j in 0..<n {
                if matrix[i][j] == 0 && !rowCovered[i] && !colCovered[j] {
                    starred[i][j] = true
                    rowCovered[i] = true
                    colCovered[j] = true
                }
            }
        }

        // Reset covers
        rowCovered = [Bool](repeating: false, count: n)
        colCovered = [Bool](repeating: false, count: n)

        // Main loop
        var maxIterations = n * n * 2  // Safety limit
        while maxIterations > 0 {
            maxIterations -= 1

            // Cover columns with starred zeros
            for i in 0..<n {
                for j in 0..<n {
                    if starred[i][j] {
                        colCovered[j] = true
                    }
                }
            }

            // Check if done
            let coveredCount = colCovered.filter { $0 }.count
            if coveredCount >= n {
                break
            }

            // Find uncovered zeros and prime them
            var found = false
            var primeRow = -1
            var primeCol = -1

            outer: for i in 0..<n {
                for j in 0..<n {
                    if matrix[i][j] == 0 && !rowCovered[i] && !colCovered[j] {
                        primed[i][j] = true
                        primeRow = i
                        primeCol = j

                        // Check if there's a starred zero in this row
                        var starCol = -1
                        for k in 0..<n {
                            if starred[i][k] {
                                starCol = k
                                break
                            }
                        }

                        if starCol >= 0 {
                            rowCovered[i] = true
                            colCovered[starCol] = false
                        } else {
                            found = true
                            break outer
                        }
                    }
                }
            }

            if found {
                // Step 5: Augmenting path
                var path = [(primeRow, primeCol)]
                var currentRow = primeRow
                var currentCol = primeCol

                while true {
                    // Find starred zero in column
                    var starRow = -1
                    for i in 0..<n {
                        if starred[i][currentCol] {
                            starRow = i
                            break
                        }
                    }

                    if starRow < 0 { break }
                    path.append((starRow, currentCol))
                    currentRow = starRow

                    // Find primed zero in row
                    var primeCol2 = -1
                    for j in 0..<n {
                        if primed[currentRow][j] {
                            primeCol2 = j
                            break
                        }
                    }

                    if primeCol2 < 0 { break }
                    path.append((currentRow, primeCol2))
                    currentCol = primeCol2
                }

                // Toggle starred and primed along path
                for (r, c) in path {
                    starred[r][c] = !starred[r][c]
                }

                // Clear primes and covers
                primed = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
                rowCovered = [Bool](repeating: false, count: n)
                colCovered = [Bool](repeating: false, count: n)

            } else {
                // Step 6: Find minimum uncovered value
                var minVal: Float = .infinity
                for i in 0..<n {
                    for j in 0..<n {
                        if !rowCovered[i] && !colCovered[j] {
                            minVal = min(minVal, matrix[i][j])
                        }
                    }
                }

                // Add to covered rows, subtract from uncovered columns
                for i in 0..<n {
                    for j in 0..<n {
                        if rowCovered[i] {
                            matrix[i][j] += minVal
                        }
                        if !colCovered[j] {
                            matrix[i][j] -= minVal
                        }
                    }
                }
            }
        }

        // Extract assignments
        var assignments: [(row: Int, col: Int)] = []
        for i in 0..<numRows {
            for j in 0..<numCols {
                if starred[i][j] {
                    assignments.append((i, j))
                }
            }
        }

        return assignments
    }
}

// MARK: - Connected Component Labeling

/// Connected component labeling for signal probability maps
struct ConnectedComponentLabeler {

    /// Label connected components in a binary mask
    /// - Parameters:
    ///   - mask: Binary mask (1 = foreground, 0 = background)
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Label array where each pixel has its component ID (0 = background)
    static func label(mask: [Float], width: Int, height: Int, threshold: Float = 0.1) -> [Int] {
        var labels = [Int](repeating: 0, count: width * height)
        var currentLabel = 0

        // BFS for each unlabeled foreground pixel
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if mask[idx] > threshold && labels[idx] == 0 {
                    currentLabel += 1
                    bfs(mask: mask, labels: &labels, startX: x, startY: y,
                        width: width, height: height, label: currentLabel, threshold: threshold)
                }
            }
        }

        return labels
    }

    private static func bfs(mask: [Float], labels: inout [Int],
                           startX: Int, startY: Int,
                           width: Int, height: Int,
                           label: Int, threshold: Float) {
        var queue: [(Int, Int)] = [(startX, startY)]
        labels[startY * width + startX] = label

        // 8-connectivity neighbors
        let dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        let dy = [-1, -1, -1, 0, 0, 1, 1, 1]

        while !queue.isEmpty {
            let (x, y) = queue.removeFirst()

            for i in 0..<8 {
                let nx = x + dx[i]
                let ny = y + dy[i]

                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let nidx = ny * width + nx
                    if mask[nidx] > threshold && labels[nidx] == 0 {
                        labels[nidx] = label
                        queue.append((nx, ny))
                    }
                }
            }
        }
    }

    /// Get bounding boxes for each component
    /// - Parameters:
    ///   - labels: Label array from label()
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Dictionary mapping label ID to (minX, minY, maxX, maxY)
    static func getBoundingBoxes(labels: [Int], width: Int, height: Int) -> [Int: (minX: Int, minY: Int, maxX: Int, maxY: Int)] {
        var boxes: [Int: (minX: Int, minY: Int, maxX: Int, maxY: Int)] = [:]

        for y in 0..<height {
            for x in 0..<width {
                let label = labels[y * width + x]
                if label > 0 {
                    if var box = boxes[label] {
                        box.minX = min(box.minX, x)
                        box.minY = min(box.minY, y)
                        box.maxX = max(box.maxX, x)
                        box.maxY = max(box.maxY, y)
                        boxes[label] = box
                    } else {
                        boxes[label] = (x, y, x, y)
                    }
                }
            }
        }

        return boxes
    }
}
