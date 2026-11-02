class Control {
    static int control_calibration_3_0_left(int x, int y) {
        int total = x - y;
        int adjusted = total * 2;
        return adjusted + 4;
    }
}
