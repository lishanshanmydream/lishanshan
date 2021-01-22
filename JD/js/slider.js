class Slider{

    constructor(id){
        // 以box为查找器 可以找到里面的所有内容
        this.box = document.querySelector(id)
        this.picBox = this.box.querySelector("ul")
        this.indexBox = this.box.querySelector(".index-box") //注意这里要加. 

        this.sliderWidth = this.box.clientWidth //一个相框的宽度

        this.index = 1
        this.sliders = this.picBox.children.length
        
        this.flag = false //处于动画中 点按钮就不响应
        this.init()

        
    }

    init(){
        console.log("slider")
        this.initPoint()
        this.copyPic()
        this.leftRight()
   //     this.play()
    }

    // 初始化小圆点
    initPoint() {
        const num = this.picBox.children.length;

        // 小圆点是dom片段
        let frg = document.createDocumentFragment();

        for(let i = 0;i<num ; i++ ){
            let li = document.createElement("li") //生成li节点
            // 在每个li上增加属性 索引值 .

            // 在生成小圆点的时候就增加一个属性。属性的值即为小圆点的值
            li.setAttribute("data-index",i+1)

            // 默认第一个圆点点亮
            if (i==0) li.className = "active"

            // 将生成的dom节点 append到片段里，最终插入时，一次性插入即可。
            frg.appendChild(li)
        }

        // 增加样式 将设置小圆点间的间距  在ol上给里制定的宽度 相当于在ol后面动态添加width = 100px
        this.indexBox.children[0].style.width = num * 10 *2 + "px"
        // 此时位置还是不够靠左，原始是有浏览器默认添加的padding 所以reset中加入对ol的初始化

        // 插入到index.html中 index—box 的 ol里
        this.indexBox.children[0].appendChild(frg)


        // 小圆点上的操作： 可以点
        this.indexBox.children[0].addEventListener(
            "click", (e) => {
                console.log("point")
                // 上面定义的小圆点的索引值 获取这个自定义属性 
                //e.target 就是获取点中的小圆点的dom节点
                let pointIndex = (e.target).getAttribute("data-index")

                //随轮播图动 索引动    当前索引值 相距之前的相差多少 
                let offset = (pointIndex - this.index) * this.sliderWidth
                this.move(offset) //点圆点可以是跳着 转换轮播图

                this.index = pointIndex //属性值拿出来
                
                console.log(this.index)
            }
        )


    }

    // 轮播图中 辅助图方案
    copyPic() {
        
        // 在1左侧增加5的辅助图，5后面增加1的辅助图， 当图片向右移动到辅助图5，将真的5替换辅助图 向做移动到辅助图1 再将真的1替换辅助图1
        
        //复制第一个图和最后一个图
        const first = this.picBox.firstElementChild.cloneNode(true) //复制第一个li元素
        const last = this.picBox.lastElementChild.cloneNode(true)

        this.picBox.appendChild(first) //最后一个元素后面插入第一张图
        this.picBox.insertBefore(last,this.picBox.firstElementChild)
        //把5放在第一个元素前面

        // 但把第5号图片方第一个了。 所以把css中的 ul的初始位置改为-590 修改tips备注

        // 这里也可以使用另一种方法，用js修改整个left的值

        // 整个ul的宽度 是一个skider的宽度 * ul里的个数
        this.picBox.style.width = this.skiderWith * this.picBox.children.length
        this.picBox.style.left = -1 * this.sliderWidth + "px"

    }

    // 轮播部分
    animate(offset){
        const time = 1000 // 切换在多长时间内完成，比如1到第2个图用 1s钟 也就是1000 ms 
        const rate = 100 //图片慢慢慢慢移动的，每移动一次用100ms
        let speed = offset/(time/rate) //计算速度 每秒移动的位移   即运动一次的位移
        // speed 是带方向的
        self.flag = true
        // 目标距离  parseFloat 得到不带单位的 算数值 （去掉单位）
        let goal = parseFloat(this.picBox.style.left) - offset //目标位置 当前的left值 - offset的值 就是目标的值
        // 由于我们patision 是abstract的 所以变动的就是 elements 里的 style 中的left的值


        // setInterval 重复rate时间做一个事情
        let animate = setInterval(()=>{ 
            // 达到目标位置goal 或非常接近(left和目标位置 距离小于移动一次)的时候不动,，没有达到的就持续动 
            if( this.picBox.style.left == goal ||  Math.abs(Math.abs(parseFloat(this.picBox.style.left)) - Math.abs(goal)) < Math.abs(speed) ){
                this.picBox.style.left = goal;
                 clearInterval(animate) //达到目的地 不需要循环操作了
                 self.flag = false

                 //复制图替换
                 if(parseFloat(this.picBox.style.left ==0)){
                    this.picBox.style.left = -this.sliders * this.sliderWidth + "px"
                 }else if (parseFloat(this.picBox.style.left) ==-(this.sliders + 1) * this.sliderWidth + "px" ){
                    this.picBox.style.left = -this.sliderWidth + "px"
                 }
            }else{
                this.picBox.style.left = parseFloat(this.picBox.style.left) - speed + "px"
            }
        },rate)

    }

    move(offset){//offset 移动距离 
        this.animate(offset) //缓动的动画效果
    
        // 索引也是跟着动的
        const num = this.indexBox.children[0].children.length //有多少个小点

        for(let i=0; i< num ; i++){
            this.indexBox.children[0].children[i].className = "" //ol里面是不同的li 就是上面生成的li  将所有active抹掉
        
        }

        // 确定当前小圆点，设置active
        this.indexBox.children[0].children[this.index-1].className = "active"
    }

    // 实现左右轮播切换
    leftRight(){
        // 选取左右箭头的dom节点
        this.box.querySelector(".left-box").addEventListener(
            "click",() => {
                console.log("left")

                if(this.flag){
                    return //处于对话中 就不移动
                }

                if(this.index - 1 < 1){
                    this.index = this.sliders //如果越界 有几个图就复制层几
                } else {
                    this.index--
                }

                this.move(-this.sliderWidth) // 点一下移动一个轮播图的宽度  向右侧移动是 正数  左侧是负数

            }
        )

        this.box.querySelector(".right-box").addEventListener(
            "click",() => {
                console.log("right")


                if(this.animated){
                    return //处于对话中 就不移动
                }

                if(this.index + 1 > this.sliders ){
                    this.index = 1 //如果越界 有几个图就复制层几
                } else {
                    this.index++
                }
                this.move(this.sliderWidth)

            }
        )
    }

    play() {

        this.auto = setInterval( () => {
            this.box.querySelector(".right-box").click()
        } ,2000)

        // 鼠标移入终止自动播放
        this.box.addEventListener("mouseenter",() => {
            clearInterval(this.auto)
        })

        //鼠标移出重新播放
        this.box.addEventListener("mouseleave", () => {
            this.auto = setInterval( () => {
                this.box.querySelector(".right-box").click()
            } ,2000)
        })
    }
}