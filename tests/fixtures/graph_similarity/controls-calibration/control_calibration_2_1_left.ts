function control_calibration_2_1_left(x: number, y: number): number {
    const total = x - y;
    const adjusted = total * 2;
    return adjusted + 4;
}
