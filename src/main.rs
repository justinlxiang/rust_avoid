use linfa::dataset::{DatasetBase, Labels};
use linfa::traits::*;
use linfa_clustering::Dbscan;
use ndarray::{Array2, Axis};
use rplidar_drv::{RplidarDevice, ScanOptions};
use std::collections::HashMap;
use reqwest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize RPLidar
    // Note: Replace "/dev/ttyUSB0" with your actual port
    // Windows example: "COM3"
    let serial_port = serialport::new("/dev/tty.usbserial-0001", 115200)
        .open()
        .unwrap();
    let mut lidar = RplidarDevice::with_stream(serial_port);

    // Stop any existing scan
    lidar.stop()?;

    // Get device info
    let info = lidar.get_device_info()?;
    println!("Connected to RPLidar device:");
    println!("  Model: {}", info.model);
    println!("  Firmware: {}", info.firmware_version as u16);
    println!("  Hardware: {}", info.hardware_version as u16);
    println!(
        "  Serial number: {}, {}",
        info.serialnum[0], info.serialnum[1]
    );

    // Start scanning
    let scan_options = ScanOptions::default();
    lidar.start_scan_with_options(&scan_options)?;

    let server_url = "http://your-ground-server.com/lidar-data"; // Replace with your actual server URL

    loop {
        // Collect points from one complete scan
        let mut scan_points = Vec::new();

        if let Ok(scan) = lidar.grab_scan() {
            for point in scan {
                // Convert polar coordinates (angle, distance) to Cartesian (x, y)
                let angle_rad = point.angle().to_radians();
                let x = point.distance() * angle_rad.cos();
                let y = point.distance() * angle_rad.sin();
                scan_points.push([x, y]);
            }
        } else {
            println!("Failed to grab scan");
            continue;
        }

        // Convert scan points to ndarray
        let points = Array2::from_shape_vec(
            (scan_points.len(), 2),
            scan_points.iter().flat_map(|p| vec![p[0], p[1]]).collect(),
        )?;

        // Create dataset
        let observations: DatasetBase<_, _> = DatasetBase::from(points);

        // Perform DBSCAN clustering
        let min_points = 3;
        let clusters = Dbscan::params(min_points)
            .tolerance(100.0) // Adjusted for LiDAR data scale
            .transform(observations)
            .unwrap();

        let label_count = clusters.label_count().remove(0);
        summarize_clusters(&label_count);

        let points = clusters.records();
        let labels = clusters.targets();

        // Create a Vec to store points for each cluster
        let num_clusters = label_count.len();
        let mut cluster_points: Vec<(usize, Array2<f32>)> = Vec::with_capacity(num_clusters);

        // Initialize empty arrays for each cluster
        for i in 0..num_clusters {
            cluster_points.push((i, Array2::zeros((0, 2))));
        }

        // Group points by their cluster labels
        for (point, &label) in points.rows().into_iter().zip(labels.iter()) {
            if let Some(cluster_id) = label {
                let point_vec = point.to_vec();
                let new_row = Array2::from_shape_vec((1, 2), point_vec).unwrap();

                let (_, arr) = &mut cluster_points[cluster_id];
                *arr = ndarray::concatenate![Axis(0), arr.view(), new_row.view()];
            }
        }

        // Calculate and store bounding box for each cluster
        let mut bounding_boxes: Vec<(usize, BoundingBox)> = Vec::new();
        for (label, points_array) in cluster_points.iter() {
            let bbox = calculate_bounding_box(points_array);
            println!("Bounding box for cluster {}: {:?}", label, bbox);
            bounding_boxes.push((*label, bbox));
        }

        // Convert labels to Vec<Option<usize>>
        let point_labels: Vec<Option<usize>> = labels.iter().map(|&l| l).collect();

        if let Err(e) = send_data_to_ground_server(&scan_points, &bounding_boxes, &point_labels, server_url).await {
            eprintln!("Failed to send data to ground server: {}", e);
        }
    }
}

fn summarize_clusters(label_count: &HashMap<Option<usize>, usize>) {
    println!("Result: ");
    for (label, count) in label_count {
        match label {
            None => println!(" - {} noise points", count),
            Some(i) => println!(" - {} points in cluster {}", count, i),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
struct BoundingBox {
    center: (f32, f32),
    width: f32,
    height: f32,
    theta: f32,
}

fn calculate_bounding_box(points: &Array2<f32>) -> BoundingBox {
    let x_coords = points.column(0);
    let y_coords = points.column(1);

    let x_min = x_coords.fold(f32::INFINITY, |acc, &x| acc.min(x));
    let x_max = x_coords.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let y_min = y_coords.fold(f32::INFINITY, |acc, &y| acc.min(y));
    let y_max = y_coords.fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));

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

#[derive(serde::Serialize)]
struct LidarData<'a> {
    timestamp: u64,
    scan_points: &'a Vec<[f32; 2]>,
    point_labels: &'a Vec<Option<usize>>,
    bounding_boxes: &'a Vec<(usize, BoundingBox)>,
}

async fn send_data_to_ground_server(
    scan_points: &Vec<[f32; 2]>,
    bounding_boxes: &Vec<(usize, BoundingBox)>,
    point_labels: &Vec<Option<usize>>,
    server_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    
    let data = LidarData {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        scan_points,
        point_labels,
        bounding_boxes,
    };

    client
        .post(server_url)
        .json(&data)
        .send()
        .await?;

    Ok(())
}