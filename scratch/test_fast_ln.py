import mlx.core as mx
try:
    import mlx.core.fast as mxf
    print("mlx.core.fast imported successfully!")
    x = mx.random.normal((4, 1024, 4096))
    g = mx.ones((4096,))
    b = mx.zeros((4096,))
    out = mxf.layer_norm(x, g, b, 1e-5)
    mx.eval(out)
    print("mx.fast.layer_norm executed successfully, shape:", out.shape)
except Exception as e:
    print("Error:", e)
