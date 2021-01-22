// 使用es6语法
class Menu {

    // 构造函数 id是分类菜单栏对应的dom节点
    constructor(id){
        // box是分类菜单栏的盒子 然后在index.html 中menu-box 定义一个id :  class="menu-box" id="menu" 这个id给js组件用的
        this.box = document.querySelector(id)
        this.ul = this.box.querySelector("ul") //获取下面的ul li submenu
        this.lis = this.box.querySelectorAll("li")  //注意这里要是querySelectorAll 才是所有的li，如果是querySelect 只返回第一个li则不是数组
        this.subMenuEles = this.box.querySelectorAll("sub-menu") 

        //不像hover那样那么快，所以定义一个定时器
        this.timer1  = null
        this.time2 = null

        this.init()
    }

    init(){
        console.log("menu")

        // 过一段时间再变，留出鼠标滑动时间 所以设置定时器。 监听时间，li 相当于一级导航菜单
        // 给每个li都绑定 mouseenter mouseleave的事件 
        // 每个item是一个li

        this.lis.forEach( (item) => {
            item.addEventListener("mouseenter",(e) => {
                let li = e.target //鼠标hover到哪个li上  就通过target获取到哪个li
                console.log("mouseenter")

                //不同的submenu 可能会触发多个timer 只留最后一个  防抖
                if(this.timer1 != null){
                    clearTimeout(this.timer1)
                }

                this.timer1 = setTimeout( () => {
                    this.subMenuEles.forEach(
                        (item) => {
                           item.classList.romove("active")
                        }
                    ) //将所有sub-menu的都去掉 active

                    //将这个li对应的sub-menu设置active
                    li.children[1].classList.add("active")
                } , 200) //每隔200ms 控制active

            })
        });




        //e代表事件
        this.lis.forEach((item)=>{
            item.addEventListener("mouseleave",(e) => {
                let li = e.target
                if (this.time2 != null) {
                    clearTimeout(this.time2)
                }
                console.log("mouseleave")
                li.children[1].classList.remove("active")
            })
        })
        
    }
}