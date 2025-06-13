const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  productionSourceMap: false,
  
  configureWebpack: {
    optimization: {
      splitChunks: {
        chunks: 'all',
        maxInitialRequests: 10,
        maxAsyncRequests: 10,
        cacheGroups: {
          // Default vendor chunk - keep it smaller
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true
          },
          
          // Core Vue libraries
          vue: {
            name: 'chunk-vue',
            test: /[\\/]node_modules[\\/](vue|@vue|vue-router|vuex)[\\/]/,
            priority: 20,
            chunks: 'initial'
          },
          
          // UI libraries (if you're using any)
          ui: {
            name: 'chunk-ui',
            test: /[\\/]node_modules[\\/](element-ui|element-plus|vuetify|quasar|ant-design-vue|bootstrap-vue)[\\/]/,
            priority: 15,
            chunks: 'initial'
          },
          
          // HTTP libraries
          http: {
            name: 'chunk-http',
            test: /[\\/]node_modules[\\/](axios|fetch|request)[\\/]/,
            priority: 15,
            chunks: 'initial'
          },
          
          // Utility libraries
          utils: {
            name: 'chunk-utils',
            test: /[\\/]node_modules[\\/](lodash|moment|date-fns|uuid|crypto-js)[\\/]/,
            priority: 15,
            chunks: 'initial'
          },
          
          // Common vendor libraries
          vendor: {
            name: 'chunk-vendors',
            test: /[\\/]node_modules[\\/]/,
            priority: -10,
            chunks: 'initial',
            maxSize: 244000 // ~240KB per chunk
          }
        }
      }
    }
  },

  chainWebpack: config => {
    // Remove prefetch for better initial load
    config.plugins.delete('prefetch')
    
    // Keep preload for critical resources
    config.plugin('preload').tap(() => [
      {
        rel: 'preload',
        include: 'initial',
        fileBlacklist: [/\.map$/, /hot-update\.js$/]
      }
    ])
    
    // Optimize images
    config.module
      .rule('images')
      .test(/\.(gif|png|jpe?g|svg)$/i)
      .use('image-webpack-loader')
      .loader('image-webpack-loader')
      .options({
        disable: process.env.NODE_ENV === 'development'
      })
      .end()
  }
})
