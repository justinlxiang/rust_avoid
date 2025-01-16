use linfa::traits::*;
use linfa_clustering::Dbscan;
use linfa_datasets::generate;
use linfa::dataset::{DatasetBase, Labels};
use ndarray::{Array2, Axis, array};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    let seed = 42;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    // `expected_centroids` has shape `(n_centroids, n_features)`
    // i.e. three points in the 2-dimensional plane
    let expected_centroids = array![[0., 0.], [-100., -100.], [100., 100.]];
    // Let's generate a synthetic dataset: three blobs of observations
    // (100 points each) centered around our `expected_centroids`
    let observations: DatasetBase<_, _> = generate::blobs(10, &expected_centroids, &mut rng).into();

    println!("Observations: {:?}", observations);

    let min_points = 3;
    let clusters = Dbscan::params(min_points)
        .tolerance(10.0)
        .transform(observations)
        .unwrap();


    println!("Clusters: {:?}", clusters);

    let label_count = clusters.label_count().remove(0);

    println!();
    println!("Result: ");
    for (label, count) in &label_count {
        match label {
            None => println!(" - {} noise points", count),
            Some(i) => println!(" - {} points in cluster {}", count, i),
        }
    }
    println!();

    // Print each point and its cluster label
    let points = clusters.records();
    let labels = clusters.targets();
    
    println!("Points and their clusters:");
    for (point, &label) in points.rows().into_iter().zip(labels.iter()) {
        let point_vec: Vec<f64> = point.to_vec();
        match label {
            None => println!("Point {:?} is noise", point_vec),
            Some(cluster_id) => println!("Point {:?} belongs to cluster {}", point_vec, cluster_id),
        }
    }

    // Create a Vec to store points for each cluster
    let num_clusters = label_count.len();
    let mut cluster_points: Vec<(usize, Array2<f64>)> = Vec::with_capacity(num_clusters);
    
    // Initialize empty arrays for each cluster
    for i in 0..num_clusters {
        cluster_points.push((i, Array2::zeros((0, 2))));
    }

    // Group points by their cluster labels 
    for (point, &label) in points.rows().into_iter().zip(labels.iter()) {
        if let Some(cluster_id) = label {
            let point_vec = point.to_vec();
            let new_row = Array2::from_shape_vec((1, 2), point_vec).unwrap();
            
            if let Some((_, arr)) = cluster_points.iter_mut().find(|(id, _)| *id == cluster_id) {
                *arr = ndarray::concatenate![Axis(0), arr.view(), new_row.view()];
            }
        }
    }

    // Calculate and store bounding box for each cluster
    let mut bounding_boxes: Vec<(usize, BoundingBox)> = Vec::new();
    for (label, points_array) in &cluster_points {
        let bbox = calculate_bounding_box(points_array);
        bounding_boxes.push((*label, bbox.clone()));
        println!("Bounding box for cluster {}: {:?}", label, bbox);
    }
}


#[derive(Debug, Clone)]
struct BoundingBox {
    center: (f64, f64),
    width: f64,
    height: f64,
    theta: f64,
}

fn calculate_bounding_box(points: &Array2<f64>) -> BoundingBox {
    let x_coords = points.column(0);
    let y_coords = points.column(1);
    
    let x_min = x_coords.fold(f64::INFINITY, |acc, &x| acc.min(x));
    let x_max = x_coords.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let y_min = y_coords.fold(f64::INFINITY, |acc, &y| acc.min(y));
    let y_max = y_coords.fold(f64::NEG_INFINITY, |acc, &y| acc.max(y));

    // Calculate bounding box properties
    let center_x = (x_min + x_max) / 2.0;
    let center_y = (y_min + y_max) / 2.0;
    let width = x_max - x_min;
    let height = y_max - y_min;
    let theta = 0.0; // Assume no rotation initially

    BoundingBox {
        center: (center_x, center_y),
        width,
        height,
        theta,
    }
}