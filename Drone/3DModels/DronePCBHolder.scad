// Parameters
arm_length = sqrt(2) * 34/2;     // Length from center to end of each arm
hole_dist = sqrt(2) * 30.5/2;
arm_width = 7.5;       // Width of each arm
arm_thickness = 2;   // Thickness of the X
hole_diameter = 2;
hole_depth = arm_thickness + 1;


module beam_between(p1, p2, width, thickness) {
    // Calculate the direction vector
    dx = p2[0] - p1[0];
    dy = p2[1] - p1[1];
    angle = atan2(dy, dx);
    len = sqrt(dx*dx + dy*dy);

    translate(p1-[0,0,thickness/2])
        rotate([0, 0, angle])
            linear_extrude(height = thickness)
                translate([0, -width/2])
                    square([len, width], center = false);
}

module extendedBeam(p1, p2, width, thickness, extension) {
     dx = p2[0] - p1[0];
    dy = p2[1] - p1[1];
    len = sqrt(dx*dx + dy*dy);

    // Normalize the direction vector
    ux = dx / len;
    uy = dy / len;

    // Extend p1 backward and p2 forward
    p1_ext = [p1[0] - ux * extension / 2, p1[1] - uy * extension / 2, p1[2]];
    p2_ext = [p2[0] + ux * extension / 2, p2[1] + uy * extension / 2, p2[2]];

    // Call the original beam module
    beam_between(p1_ext, p2_ext, width, thickness);
}

module hole(p1, hole_diameter, hole_depth) {
    translate(p1)
    cylinder(h = hole_depth, d = hole_diameter, center = true, $fn = 20);
}

module hole_custom_sides(p1, hole_diameter, hole_depth, side_num) {
    translate(p1)
    cylinder(h = hole_depth, d = hole_diameter, center = true, $fn = side_num);
}

// Main X shape with holes
difference() {
    union() {
        extendedBeam([30.5/2,30.5/2,0], [32.8,23,0], width=arm_width, thickness = arm_thickness, extension=7);
        extendedBeam([-30.5/2,30.5/2,0], [-32.8,23,0], width=arm_width, thickness = arm_thickness, extension=7);
        extendedBeam([-30.5/2,-30.5/2,0], [-32.8,-23,0], width=arm_width, thickness = arm_thickness, extension=7);
        extendedBeam([30.5/2,-30.5/2,0], [32.8,-23,0], width=arm_width, thickness = arm_thickness, extension=7);
        
        extendedBeam([30.5/2,-30.5/2,0], [30.5/2,30.5/2,0], width=arm_width, thickness = arm_thickness, extension=0);
        extendedBeam([-30.5/2,-30.5/2,0], [-30.5/2,30.5/2,0], width=arm_width, thickness = arm_thickness, extension=0);
        
        
        extendedBeam([-32.8,-23,0], [32.8,-23,0], width=arm_width, thickness = arm_thickness, extension=0);
        extendedBeam([-32.8,23,0], [32.8,23,0], width=arm_width, thickness = arm_thickness, extension=0);
        // First arm (diagonal 45°)
        rotate([0, 0, 45])
            cube([arm_length * 2, arm_width, arm_thickness], center = true);
        
        // Second arm (diagonal -45°)
        rotate([0, 0, -45])
            cube([arm_length * 2, arm_width, arm_thickness], center = true);
    }

    hole([32.8,23,0], hole_diameter, hole_depth);
    hole([-32.8,23,0], hole_diameter, hole_depth);
    hole([-32.8,-23,0], hole_diameter, hole_depth);
    hole([32.8,-23,0], hole_diameter, hole_depth);
    // Holes at the ends of the X
    for (angle = [45, 135, 225, 315]) {
        rotate([0, 0, angle])
            translate([hole_dist, 0, 0])
                cylinder(h = hole_depth, d = hole_diameter, center = true, $fn = 20);
    }
    
    hole_custom_sides([-32.8,23,1],5, 2, 6);
    hole_custom_sides([-32.8,-23,1],5, 2, 6);
    hole_custom_sides([32.8,-23,1],5, 2, 6);
    hole_custom_sides([32.8,23,1],5, 2, 6);
    
    hole_custom_sides([30.5/2,30.5/2,-1],5, 2, 6);
    hole_custom_sides([-30.5/2,30.5/2,-1],5, 2, 6);
    hole_custom_sides([30.5/2,-30.5/2,-1],5, 2, 6);
    hole_custom_sides([-30.5/2,-30.5/2,-1],5, 2, 6);
}
