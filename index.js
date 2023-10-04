const API_URL = 'http://muds.gdl.jp/s2122027/'
fetch(API_URL)
    .then(response => response.json())
    .then(data => {
        console.log(data)
    }
)
