let query = {"inputs": {"context_id": [3, 5, 8, 0, 0, 0, 0, 0, 0, 0], "context_rating": [3.0, 3.0, 3.0, 0, 0, 0, 0, 0, 0, 0]}}

let test = async () => await fetch ('http://localhost:8501/v1/models/wrecksys:predict', {
    method: 'GET'
})

const caller = async () => {
    const json = await test()
    console.log(json)
}


/**
let resp = async () => await fetch('http://localhost:8501/v1/models/wrecksys:serve', {
    method: 'POST',
    body: JSON.stringify(query)
  }).then(async (response) => {
      let t= await response.json()
      console.log(t)
      return t
    })
    .then((results) => results[1].array())
**/

// console.log(resp())

