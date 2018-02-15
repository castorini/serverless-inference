const async = require('async');
const fs = require('fs');
const request = require('request');
const _ = require('lodash');
const readline = require('readline');

if (process.argv.length != 4) {
    console.log('Usage: node query_sm_model.js API_URL num_items');
    process.exit(1);
}

var url = process.argv[2];
var num_items = process.argv[3];

function readFile(path, cb) {
    var data = [];

    const rl = readline.createInterface({
        input: fs.createReadStream(path)
    });

    rl.on('line', line => data.push(line));
    rl.on('close', () => cb(null, data));
}

const TRECQA_DEV_A = '../../data/TrecQA/raw-dev/a.toks';
const TRECQA_DEV_B = '../../data/TrecQA/raw-dev/b.toks';

async.series([
    function(callback) {
        readFile(TRECQA_DEV_A, callback);
    },
    function(callback) {
        readFile(TRECQA_DEV_B, callback);
    }
], function(err, results) {
    var trec_dev = _.zip(results[0], results[1]).slice(0, num_items);
    var size = trec_dev.length;
    console.log(`TrecQA dev set loaded (${size} sentences)`);
    var t0 = process.hrtime();
    var latencies = [];

    async.map(trec_dev, function(qa_pair, callback) {
        var request_start = process.hrtime();
        request.post(
            url,
            {
                body: JSON.stringify({
                    sent1: qa_pair[0],
                    sent2: qa_pair[1]
                })
            },
            function(error, response, data) {
                if (!error) {
                    var since_request_start = process.hrtime(request_start);
                    var request_elapsed = since_request_start[0] + since_request_start[1] / 1000000000;
                    latencies.push(request_elapsed);
                    callback(null, data);
                } else {
                    console.err('ERROR!');
                    callback('error', data);
                }
            }
        );
    },
    function(err, results) {
        console.log(results);
        var since_t0 = process.hrtime(t0);
        var elapsed = since_t0[0] + since_t0[1] / 1000000000;
        console.log('========================================');
        console.log(`${size} queries took ${elapsed} ms. Throughput is ${size / elapsed} qps.`);

        latencies.sort();
        var p50 = latencies[Math.floor(latencies.length * 0.5) - 1];
        var p99 = latencies[Math.floor(latencies.length * 0.99) - 1];
        var avg = _.mean(latencies);
        console.log(`${latencies.length} requests successfully returned. p50 latency is ${p50} s, p99 latency is ${p99}, avg. latency is ${avg}.`);
    });
});

