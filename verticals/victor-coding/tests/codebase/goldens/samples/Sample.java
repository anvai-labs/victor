public class Sample {
    private int count = 0;

    public int increment() {
        count = count + 1;
        return count;
    }

    protected String format(String input) {
        if (input == null) {
            return "";
        }
        return input.trim();
    }
}
