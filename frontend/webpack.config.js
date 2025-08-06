const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: "./src/index.js",             // where webpack starts
  output: {
    path: path.resolve(__dirname, "build"),
    filename: "bundle.[contenthash].js",
    clean: true,                       // delete old builds
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,           // any .js or .jsx file
        exclude: /node_modules/,
        use: "babel-loader",           // run through Babel
      },
    ],
  },
  resolve: { extensions: [".js", ".jsx"] },
  plugins: [
    new HtmlWebpackPlugin({
      template: "public/index.html",   // use your HTML as template
    }),
  ],
  devServer: {
    port: 3000,
    hot: true,
    historyApiFallback: true,
  },
};
