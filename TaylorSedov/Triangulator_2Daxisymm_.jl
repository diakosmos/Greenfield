using Plots
using Random

# 2D Unstructured Mesh Generator for CFD
# Creates triangular meshes suitable for finite-volume calculations

# Mesh data structures
mutable struct Node
    id::Int
    x::Float64
    y::Float64
    boundary::Bool  # Is this a boundary node?
end

mutable struct Face
    id::Int
    node1::Int      # First node ID
    node2::Int      # Second node ID
    cell_left::Int  # Left cell ID (0 if boundary)
    cell_right::Int # Right cell ID (0 if boundary)
    boundary::Bool  # Is this a boundary face?
    normal_x::Float64  # Outward normal x-component
    normal_y::Float64  # Outward normal y-component
    area::Float64      # Face length (in 2D)
    center_x::Float64  # Face center x
    center_y::Float64  # Face center y
end

mutable struct Cell
    id::Int
    node1::Int      # First node ID
    node2::Int      # Second node ID  
    node3::Int      # Third node ID
    face1::Int      # First face ID
    face2::Int      # Second face ID
    face3::Int      # Third face ID
    center_x::Float64  # Cell centroid x
    center_y::Float64  # Cell centroid y
    area::Float64      # Cell area
    volume::Float64    # Cell volume (for axisymmetric: 2π ∫ r dA)
end

mutable struct Mesh
    nodes::Vector{Node}
    faces::Vector{Face}
    cells::Vector{Cell}
    n_nodes::Int
    n_faces::Int
    n_cells::Int
    
    # Boundary information
    boundary_faces::Vector{Int}  # List of boundary face IDs
    
    # Domain bounds
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
end

# Create empty mesh
function Mesh()
    return Mesh(Node[], Face[], Cell[], 0, 0, 0, Int[], 0.0, 0.0, 0.0, 0.0)
end

# Point-in-triangle test (used for Delaunay triangulation)
function point_in_triangle(px::Float64, py::Float64, 
                          x1::Float64, y1::Float64,
                          x2::Float64, y2::Float64, 
                          x3::Float64, y3::Float64)
    # Using barycentric coordinates
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12
        return false
    end
    
    a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    c = 1 - a - b
    
    return a >= -1e-12 && b >= -1e-12 && c >= -1e-12
end

# Circumcircle test for Delaunay triangulation
function in_circumcircle(px::Float64, py::Float64,
                        x1::Float64, y1::Float64,
                        x2::Float64, y2::Float64,
                        x3::Float64, y3::Float64)
    # Check if point (px, py) is inside circumcircle of triangle (x1,y1)-(x2,y2)-(x3,y3)
    ax = x1 - px
    ay = y1 - py
    bx = x2 - px  
    by = y2 - py
    cx = x3 - px
    cy = y3 - py
    
    det = (ax*ax + ay*ay) * (bx*cy - by*cx) +
          (bx*bx + by*by) * (cx*ay - cy*ax) +
          (cx*cx + cy*cy) * (ax*by - ay*bx)
    
    return det > 1e-12
end

# Simple Delaunay triangulation using Bowyer-Watson algorithm
function delaunay_triangulation(points::Vector{Tuple{Float64, Float64}})
    n = length(points)
    if n < 3
        error("Need at least 3 points for triangulation")
    end
    
    # Find bounding box and create super triangle
    x_coords = [p[1] for p in points]
    y_coords = [p[2] for p in points]
    
    x_min, x_max = extrema(x_coords)
    y_min, y_max = extrema(y_coords)
    
    dx = x_max - x_min
    dy = y_max - y_min
    delta = max(dx, dy) * 2
    
    # Super triangle vertices (large triangle containing all points)
    super1 = (x_min - delta, y_min - delta)
    super2 = (x_max + delta, y_min - delta)  
    super3 = ((x_min + x_max)/2, y_max + delta)
    
    # Initialize with super triangle
    triangles = [[(n+1, n+2, n+3)]]  # Store as node indices
    all_points = [points; [super1, super2, super3]]
    
    # Add points one by one
    for i in 1:n
        px, py = points[i]
        bad_triangles = Int[]
        polygon_edges = Tuple{Int,Int}[]
        
        # Find triangles whose circumcircle contains the point
        for (tri_idx, tri) in enumerate(triangles[1])
            i1, i2, i3 = tri
            x1, y1 = all_points[i1]
            x2, y2 = all_points[i2] 
            x3, y3 = all_points[i3]
            
            if in_circumcircle(px, py, x1, y1, x2, y2, x3, y3)
                push!(bad_triangles, tri_idx)
                # Add edges to polygon
                push!(polygon_edges, (i1, i2))
                push!(polygon_edges, (i2, i3))
                push!(polygon_edges, (i3, i1))
            end
        end
        
        # Remove bad triangles
        triangles[1] = [triangles[1][j] for j in 1:length(triangles[1]) if !(j in bad_triangles)]
        
        # Remove duplicate edges (internal edges of the polygon)
        unique_edges = Tuple{Int,Int}[]
        for edge in polygon_edges
            count = sum(e == edge || e == (edge[2], edge[1]) for e in polygon_edges)
            if count == 1  # Edge appears only once
                push!(unique_edges, edge)
            end
        end
        
        # Create new triangles by connecting point to polygon boundary
        for edge in unique_edges
            push!(triangles[1], (edge[1], edge[2], i))
        end
    end
    
    # Remove triangles that use super triangle vertices
    final_triangles = Tuple{Int,Int,Int}[]
    for tri in triangles[1]
        i1, i2, i3 = tri
        if i1 <= n && i2 <= n && i3 <= n  # All vertices are original points
            push!(final_triangles, tri)
        end
    end
    
    return final_triangles, points
end

# Generate random points in a square domain
function generate_random_points(nx::Int, ny::Int, x_min::Float64, x_max::Float64, 
                               y_min::Float64, y_max::Float64; 
                               boundary_spacing::Float64=0.1, seed::Int=42)
    Random.seed!(seed)
    points = Tuple{Float64, Float64}[]
    
    # Add boundary points for better mesh quality
    dx_boundary = (x_max - x_min) / (nx - 1)
    dy_boundary = (y_max - y_min) / (ny - 1)
    
    # Bottom boundary
    for i in 1:nx
        x = x_min + (i-1) * dx_boundary
        push!(points, (x, y_min))
    end
    
    # Top boundary  
    for i in 1:nx
        x = x_min + (i-1) * dx_boundary
        push!(points, (x, y_max))
    end
    
    # Left boundary (excluding corners)
    for j in 2:ny-1
        y = y_min + (j-1) * dy_boundary
        push!(points, (x_min, y))
    end
    
    # Right boundary (excluding corners)
    for j in 2:ny-1
        y = y_min + (j-1) * dy_boundary  
        push!(points, (x_max, y))
    end
    
    # Interior random points
    n_interior = (nx-2) * (ny-2)
    for i in 1:n_interior
        x = x_min + boundary_spacing + (x_max - x_min - 2*boundary_spacing) * rand()
        y = y_min + boundary_spacing + (y_max - y_min - 2*boundary_spacing) * rand()
        push!(points, (x, y))
    end
    
    return points
end

# Build mesh connectivity from triangulation
function build_mesh_connectivity!(mesh::Mesh, triangles::Vector{Tuple{Int,Int,Int}}, 
                                 points::Vector{Tuple{Float64, Float64}})
    # Clear existing data
    empty!(mesh.nodes)
    empty!(mesh.faces) 
    empty!(mesh.cells)
    empty!(mesh.boundary_faces)
    
    # Create nodes
    mesh.n_nodes = length(points)
    for (i, (x, y)) in enumerate(points)
        # Check if boundary node
        is_boundary = (abs(x - mesh.x_min) < 1e-12 || abs(x - mesh.x_max) < 1e-12 ||
                      abs(y - mesh.y_min) < 1e-12 || abs(y - mesh.y_max) < 1e-12)
        push!(mesh.nodes, Node(i, x, y, is_boundary))
    end
    
    # Create cells and collect edges
    mesh.n_cells = length(triangles)
    edge_map = Dict{Tuple{Int,Int}, Vector{Int}}()  # Edge -> list of adjacent cells
    
    for (cell_id, (n1, n2, n3)) in enumerate(triangles)
        # Calculate cell center and area
        x1, y1 = points[n1]
        x2, y2 = points[n2] 
        x3, y3 = points[n3]
        
        center_x = (x1 + x2 + x3) / 3
        center_y = (y1 + y2 + y3) / 3
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        
        # Volume for axisymmetric case (2π * r * area, where r is centroid radius)
        r_centroid = sqrt(center_x^2 + center_y^2)
        volume = 2 * π * r_centroid * area
        
        # Create cell (face IDs will be filled later)
        push!(mesh.cells, Cell(cell_id, n1, n2, n3, 0, 0, 0, center_x, center_y, area, volume))
        
        # Add edges to map
        edges = [(min(n1,n2), max(n1,n2)), 
                (min(n2,n3), max(n2,n3)), 
                (min(n3,n1), max(n3,n1))]
        
        for edge in edges
            if !haskey(edge_map, edge)
                edge_map[edge] = Int[]
            end
            push!(edge_map[edge], cell_id)
        end
    end
    
    # Create faces from edges
    mesh.n_faces = 0
    face_id = 0
    
    for ((n1, n2), adjacent_cells) in edge_map
        face_id += 1
        
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        
        # Face center and length
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        area = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        
        # Normal vector (outward from left cell)
        dx = x2 - x1
        dy = y2 - y1
        normal_x = dy / area   # Rotate 90 degrees
        normal_y = -dx / area
        
        # Determine left and right cells
        is_boundary = length(adjacent_cells) == 1
        cell_left = adjacent_cells[1]
        cell_right = is_boundary ? 0 : adjacent_cells[2]
        
        # Create face
        face = Face(face_id, n1, n2, cell_left, cell_right, is_boundary,
                   normal_x, normal_y, area, center_x, center_y)
        push!(mesh.faces, face)
        
        if is_boundary
            push!(mesh.boundary_faces, face_id)
        end
        
        # Update cell face references (simplified - just store some face IDs)
        if mesh.cells[cell_left].face1 == 0
            mesh.cells[cell_left].face1 = face_id
        elseif mesh.cells[cell_left].face2 == 0
            mesh.cells[cell_left].face2 = face_id
        else
            mesh.cells[cell_left].face3 = face_id
        end
        
        if !is_boundary
            if mesh.cells[cell_right].face1 == 0
                mesh.cells[cell_right].face1 = face_id
            elseif mesh.cells[cell_right].face2 == 0
                mesh.cells[cell_right].face2 = face_id
            else
                mesh.cells[cell_right].face3 = face_id
            end
        end
    end
    
    mesh.n_faces = face_id
    
    println("Mesh created successfully:")
    println("  Nodes: $(mesh.n_nodes)")
    println("  Cells: $(mesh.n_cells)")
    println("  Faces: $(mesh.n_faces)")
    println("  Boundary faces: $(length(mesh.boundary_faces))")
end

# Main mesh generation function
function generate_mesh(nx::Int, ny::Int, x_min::Float64, x_max::Float64,
                      y_min::Float64, y_max::Float64; seed::Int=42)
    
    println("Generating 2D unstructured mesh...")
    println("Domain: [$x_min, $x_max] × [$y_min, $y_max]")
    println("Approximate resolution: $nx × $ny")
    
    # Create mesh object
    mesh = Mesh()
    mesh.x_min = x_min
    mesh.x_max = x_max
    mesh.y_min = y_min
    mesh.y_max = y_max
    
    # Generate points
    points = generate_random_points(nx, ny, x_min, x_max, y_min, y_max, seed=seed)
    println("Generated $(length(points)) points")
    
    # Perform Delaunay triangulation
    println("Performing Delaunay triangulation...")
    triangles, final_points = delaunay_triangulation(points)
    println("Created $(length(triangles)) triangles")
    
    # Build mesh connectivity
    println("Building mesh connectivity...")
    build_mesh_connectivity!(mesh, triangles, final_points)
    
    return mesh
end

# Visualization function
function plot_mesh(mesh::Mesh; show_nodes::Bool=true, show_cell_ids::Bool=false)
    # Plot triangles
    p = plot(aspect_ratio=:equal, legend=false, title="2D Unstructured Mesh")
    
    # Draw all triangles
    for cell in mesh.cells
        n1, n2, n3 = cell.node1, cell.node2, cell.node3
        x_coords = [mesh.nodes[n1].x, mesh.nodes[n2].x, mesh.nodes[n3].x, mesh.nodes[n1].x]
        y_coords = [mesh.nodes[n1].y, mesh.nodes[n2].y, mesh.nodes[n3].y, mesh.nodes[n1].y]
        plot!(p, x_coords, y_coords, color=:black, linewidth=1)
    end
    
    # Highlight boundary faces
    for face_id in mesh.boundary_faces
        face = mesh.faces[face_id]
        n1, n2 = face.node1, face.node2
        x_coords = [mesh.nodes[n1].x, mesh.nodes[n2].x]
        y_coords = [mesh.nodes[n1].y, mesh.nodes[n2].y]
        plot!(p, x_coords, y_coords, color=:red, linewidth=3)
    end
    
    # Show nodes
    if show_nodes
        x_nodes = [node.x for node in mesh.nodes]
        y_nodes = [node.y for node in mesh.nodes]
        scatter!(p, x_nodes, y_nodes, color=:blue, markersize=2)
    end
    
    # Show cell IDs
    if show_cell_ids
        for cell in mesh.cells
            if cell.id <= 20  # Only show first 20 to avoid clutter
                annotate!(p, cell.center_x, cell.center_y, text(string(cell.id), 8))
            end
        end
    end
    
    xlabel!(p, "x")
    ylabel!(p, "y")
    
    return p
end

# Test the mesh generator
println("="^60)
println("2D Unstructured Mesh Generator for CFD")
println("="^60)

# Generate a test mesh
mesh = generate_mesh(10, 8, 0.0, 1.0, 0.0, 0.8, seed=123)

# Plot the mesh
p = plot_mesh(mesh, show_nodes=true, show_cell_ids=false)
display(p)

println("\nMesh statistics:")
println("  Average cell area: $(round(sum(cell.area for cell in mesh.cells) / mesh.n_cells, digits=6))")
println("  Min cell area: $(round(minimum(cell.area for cell in mesh.cells), digits=6))")
println("  Max cell area: $(round(maximum(cell.area for cell in mesh.cells), digits=6))")

println("\nMesh data structure ready for CFD solver!")
println("Key features:")
println("✓ Nodes with boundary identification")
println("✓ Faces with connectivity and geometry") 
println("✓ Cells with area and volume (for axisymmetric)")
println("✓ Boundary face tracking")
println("✓ All connectivity information for finite-volume method")

global mesh
