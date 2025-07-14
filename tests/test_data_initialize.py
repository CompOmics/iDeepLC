# Python
import pandas as pd
from ideeplc.data_initialize import data_initialize

def test_data_initialize():
    """Test the data_initialize function."""
    test_csv_path = "ideeplc/example_input/Hela_deeprt.csv"  # Path to a sample test CSV file
    matrix_input, x_shape = data_initialize(csv_path=test_csv_path)

    df_0 = pd.DataFrame(matrix_input[0][0])
    expected = pd.DataFrame({
    1: [-0.796144, -0.941486, 10.000000, 12.000000, 4.000000, 4.0],
    2: [-0.145342, 0.958311, 11.000000, 12.000000, 2.000000, 2.0],
    3: [1.546797, 1.007078, 11.000000, 18.000000, 4.000000, 2.0],
    4: [-0.588486, -1.384630, 6.000000, 8.000000, 4.000000, 2.0]
}, index=[0, 1, 2, 3, 4, 5])

    pd.testing.assert_frame_equal(df_0.iloc[:6, 1:5], expected, check_dtype=False)
    assert matrix_input is not None, "Matrix input should not be None"
    assert isinstance(x_shape, tuple), "x_shape should be a tuple"
    assert x_shape == (1, 41, 62), "Expected shape of x is (1, 41, 62)"

