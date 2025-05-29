<template>
    <div class="v-spinner" v-show="loading">
        <div class="v-clip" v-bind:style="spinnerStyle">
        </div>
        <span class="loading-text animated-fade" :style="textStyle">{{ msg }}</span>     
    </div>
</template>

<script>
export default {
    
    name: 'ClipLoader',

    props: {
    loading: {
        type: Boolean,
        default: true
    },
        color: { 
        type: String,
        default: '#5dc596'
    },
        size: {
        type: String,
        default: '35px'
    },
    radius: {
        type: String,
        default: '100%'
    },
    msg: {
        type: String,
        default: 'Generating Model',
    },
    },
    computed: {
    spinnerStyle () {
        return {
        height: this.size,
        width: this.size,
        borderWidth: '2px',
        borderStyle: 'solid',
        borderColor: this.color + ' ' + this.color + ' transparent',
        borderRadius: this.radius,
        background: 'transparent'
        }
    },
    textStyle () {
      const numericSize = parseFloat(this.size);
      const fontSize = numericSize / 4; // Adjust this factor as needed
      return {
        fontSize: `${fontSize}px`,
        marginTop: `${numericSize / 4}px`, // Adjust margin based on size
        color: '#333',
        fontFamily: 'sans-serif',
        fontWeight: 400,
        textAlign: 'center' // Added for better alignment
      }
    }
    
    }
}
</script>

<style>

.v-spinner
{
    display: flex;
    flex-direction: column; /* Stack the spinner and text vertically */
    align-items: center; /* Center them horizontally */
    text-align: center; /* Center the text */
    
}

.animated-fade {
  animation: fadeInOut 0.75s infinite alternate;
}

@keyframes fadeInOut {
  0% { opacity: 0; }
  100% { opacity: 1; }
}

.v-spinner .v-clip
{
    -webkit-animation: v-clipDelay 0.75s 0s infinite linear;
            animation: v-clipDelay 0.75s 0s infinite linear;
    -webkit-animation-fill-mode: both;
                animation-fill-mode: both;

    display: inline-block;
}

@-webkit-keyframes v-clipDelay
{
    0%
    {
        -webkit-transform: rotate(0deg);
                transform: rotate(0deg);
    }
    50%
    {
        -webkit-transform: rotate(180deg);
                transform: rotate(180deg);
    }
    100%
    {
        -webkit-transform: rotate(360deg);
                transform: rotate(360deg);
    }
}

@keyframes v-clipDelay
{
    0%
    {
        -webkit-transform: rotate(0deg);
                transform: rotate(0deg);
    }
    50%
    {
        -webkit-transform: rotate(180deg);
                transform: rotate(180deg);
    }
    100%
    {
        -webkit-transform: rotate(360deg);
                transform: rotate(360deg);
    }
}
</style>