module.exports = {
  apps: [{
    name: 'vue-frontend',
    script: 'npm',
    args: 'run serve',
    cwd: '/home/ubuntu/yourproject/frontend',
    env: {
      NODE_ENV: 'production',
      HOST: '0.0.0.0',
      PORT: 3000
    }
  }]
}
