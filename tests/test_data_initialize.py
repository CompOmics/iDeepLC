# Python
import pandas as pd
from ideeplc.data_initialize import data_initialize

def test_data_initialize():
    """Test the data_initialize function."""
    test_csv_path = "ideeplc/example_input/Hela_deeprt.csv"  # Path to a sample test CSV file
    matrix_input, x_shape = data_initialize(csv_path=test_csv_path)

    df_0 = pd.DataFrame(matrix_input[0][3])
    expected = pd.DataFrame({
        1: [1.286264, 0.209934, 12.0, 23.0, 2.0, 5.0],
        2: [-1.076330, -2.715786, 6.0, 11.0, 7.0, 2.0],
        3: [-1.435417, 1.267611, 14.0, 16.0, 4.0, 2.0],
        4: [-1.280370, -2.870834, 6.0, 10.0, 4.0, 2.0]
    }, index=[0, 1, 2, 3, 4, 5])

    pd.testing.assert_frame_equal(df_0.iloc[:6, 1:5], expected, check_dtype=False)
    assert matrix_input is not None, "Matrix input should not be None"
    assert isinstance(x_shape, tuple), "x_shape should be a tuple"
    assert x_shape == (1, 41, 62), "Expected shape of x is (1, 41, 62)"

